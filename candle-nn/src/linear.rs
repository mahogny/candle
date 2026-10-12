//! Linear layer
//!
//! This layer applies a linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{Linear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Cpu)?;
//! let layer = Linear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[10f32, 100.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[210.0, 430.0, 650.0]]);
//! # Ok(()) }
//! ```
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use rayon::prelude::*;
use std::sync::{Once, OnceLock};

static GEMM_TUNING: Once = Once::new();
static TRANSPOSE_GEMM: OnceLock<bool> = OnceLock::new();
static LINEAR_PARALLEL_SEED_MIN: OnceLock<usize> = OnceLock::new();

fn configure_gemm_from_env() {
    GEMM_TUNING.call_once(|| {
        if let Some(value) = std::env::var_os("CANDLE_GEMM_RHS_PACKING_THRESHOLD")
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
        {
            // The GEMM crate clamps this to a safe architecture-independent
            // range. This opt-in benchmark control makes packing decisions
            // attributable without changing the default backend policy.
            gemm::set_rhs_packing_threshold(value);
        }
        if let Some(value) = std::env::var_os("CANDLE_GEMM_THREADING_THRESHOLD")
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
        {
            // Attribution control for GEMM's Rayon crossover. The setting is
            // process-global in gemm, so initialize it before the first fused
            // linear call and keep the normal backend default when absent.
            gemm::set_threading_threshold(value);
        }
        if let Some(value) = std::env::var_os("CANDLE_GEMM_LHS_PACKING_THRESHOLD")
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
        {
            // Input packing is shape-dependent and appears directly in CPU
            // samples. Keep the backend defaults unless a real-model A/B
            // demonstrates that a different crossover is beneficial.
            gemm::set_lhs_packing_threshold_single_thread(value);
            gemm::set_lhs_packing_threshold_multi_thread(value);
        }
        if std::env::var_os("CANDLE_GEMM_DISABLE_AVX512").is_some_and(|value| value == "1") {
            gemm::set_avx512_enabled(false);
        }
    });
}

#[derive(Clone, Debug)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Prepares the immutable weight in contiguous `(input, output)` order for
    /// repeated CPU-f32 inference.
    ///
    /// This is deliberately explicit rather than a global hidden cache: model
    /// code can bound the extra storage to projections that are called often
    /// enough to repay it. Cloning the returned tensor is shallow, so callers
    /// can share one prepared weight across workers without copying its data.
    pub fn prepare_transposed_weight_cpu_f32(&self) -> Result<Option<Tensor>> {
        if !self.weight.device().is_cpu() || self.weight.dtype() != DType::F32 {
            return Ok(None);
        }
        Ok(Some(self.weight.t()?.contiguous()?))
    }

    /// Applies this linear layer using a weight prepared by
    /// [`Self::prepare_transposed_weight_cpu_f32`].
    ///
    /// The prepared `(input, output)` storage lets the GEMM backend consume
    /// the immutable operand in its preferred orientation without repacking
    /// the original `(output, input)` tensor on every call. Invalid layouts
    /// fall back to the ordinary fused-bias route.
    pub fn forward_fused_bias_cpu_pretransposed(
        &self,
        x: &Tensor,
        transposed_weight: &Tensor,
    ) -> Result<Tensor> {
        if let Some(bias) = &self.bias {
            if let Some(output) = linear_cpu_f32_bias_impl(
                x,
                transposed_weight,
                bias,
                None,
                false,
                false,
                false,
                true,
            )? {
                return Ok(output);
            }
        }
        self.forward_fused_bias_cpu(x)
    }

    /// Applies this linear layer and adds a same-shaped residual.
    ///
    /// The contiguous CPU f32 path seeds the final output with the residual
    /// and bias, then accumulates GEMM into it. Other layouts/devices retain
    /// the ordinary Tensor composition.
    pub fn forward_add_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if let Some(bias) = &self.bias {
            if let Some(output) = linear_cpu_f32_bias_residual(x, &self.weight, bias, residual)? {
                return Ok(output);
            }
        }
        residual.broadcast_add(&super::Module::forward(self, x)?)
    }

    /// Applies the existing contiguous CPU f32 bias fusion when its strict
    /// layout gate matches, otherwise uses the ordinary linear operation.
    pub fn forward_fused_bias_cpu(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if std::env::var_os("CANDLE_CUDA_LINEAR_BIAS").is_none_or(|value| value != "0") {
            if let Some(bias) = &self.bias {
                if let Some(output) = linear_cuda_f32_bias(x, &self.weight, bias)? {
                    return Ok(output);
                }
            }
        }
        // Keep a process-level attribution switch: SAM2 uses this explicitly
        // for lower-copy Hiera/decoder paths, and it is useful to distinguish
        // the fused GEMM from surrounding model work without changing callers.
        if std::env::var_os("CANDLE_CPU_FORCE_FUSED_BIAS").is_none_or(|value| value != "0") {
            if let Some(bias) = &self.bias {
                if let Some(output) = linear_cpu_f32_bias(x, &self.weight, bias)? {
                    return Ok(output);
                }
            }
        }
        super::Module::forward(self, x)
    }

    /// Apply a CPU f32 linear layer followed by exact GELU in the owned GEMM
    /// output, saving one activation tensor for large MLP expansions.
    pub fn forward_fused_bias_gelu_erf_cpu(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var_os("CANDLE_CPU_LINEAR_GELU_FUSION").is_none_or(|value| value != "0") {
            if let Some(bias) = &self.bias {
                if let Some(output) = linear_cpu_f32_bias_impl(
                    x,
                    &self.weight,
                    bias,
                    None,
                    true,
                    false,
                    false,
                    false,
                )? {
                    return Ok(output);
                }
            }
        }
        self.forward_fused_bias_cpu(x)?.gelu_erf()
    }

    /// Apply a CPU f32 linear layer followed by ReLU in the owned GEMM
    /// destination, avoiding a second activation-sized tensor.
    pub fn forward_fused_bias_relu_cpu(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var_os("CANDLE_CPU_LINEAR_RELU_FUSION").is_none_or(|value| value != "0") {
            if let Some(bias) = &self.bias {
                if let Some(output) = linear_cpu_f32_bias_impl(
                    x,
                    &self.weight,
                    bias,
                    None,
                    false,
                    true,
                    false,
                    false,
                )? {
                    return Ok(output);
                }
            }
        }
        self.forward_fused_bias_cpu(x)?.relu()
    }
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        // The fused CPU implementation is retained for controlled attribution,
        // but is opt-in until it wins end-to-end against Candle's established
        // matmul path for real model shapes.
        if std::env::var_os("CANDLE_CPU_LINEAR_BIAS").is_some_and(|value| value == "1") {
            if let Some(bias) = &self.bias {
                if let Some(output) = linear_cpu_f32_bias(x, &self.weight, bias)? {
                    return Ok(output);
                }
            }
        }
        // CUDA matmul requires contiguous operands. Materialize non-contiguous
        // activation views here so rank-3/4 Linear inputs can use the flattened
        // GEMM route instead of an unsupported broadcasted matmul.
        let x = if x.device().is_cuda() && !x.is_contiguous() {
            x.contiguous()?
        } else {
            x.clone()
        };
        // When possible, we avoid using a broadcasted matmul as it is much slower
        // than the standard matmul for the cuda and cpu backends.
        let x = match *x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((b1 * b2 * m, k))?
                        .matmul(&w)?
                        .reshape((b1, b2, m, ()))?
                } else {
                    let w = self.weight.broadcast_left((b1, b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = self.weight.t()?;
                    x.reshape((bsize * m, k))?
                        .matmul(&w)?
                        .reshape((bsize, m, ()))?
                } else {
                    let w = self.weight.broadcast_left(bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        match &self.bias {
            None => Ok(x),
            Some(bias) => bias_add_last_dim(&x, bias),
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct LinearCudaF32Bias;

#[cfg(feature = "cuda")]
impl candle::CustomOp3 for LinearCudaF32Bias {
    fn name(&self) -> &'static str {
        "linear-cuda-f32-bias"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("linear-cuda-f32-bias is only implemented for cuda")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        weight: &candle::CudaStorage,
        weight_l: &Layout,
        bias: &candle::CudaStorage,
        bias_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::cublaslt::{Matmul, MatmulConfig};
        use candle::cuda_backend::{CudaStorage, CudaStorageSlice};

        let (x_start, x_end) = x_l
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("linear input must be contiguous".into()))?;
        let (weight_start, weight_end) = weight_l
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("linear weight must be contiguous".into()))?;
        let (bias_start, bias_end) = bias_l
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::Msg("linear bias must be contiguous".into()))?;
        let [output_dim, input_dim] = weight_l.dims() else {
            candle::bail!("linear weight must have rank two")
        };
        let Some(&x_input_dim) = x_l.dims().last() else {
            candle::bail!("linear input must have non-zero rank")
        };
        if x_input_dim != *input_dim || bias_l.shape().elem_count() != *output_dim {
            candle::bail!("linear input, weight, and bias dimensions do not match")
        }
        let rows = x_l.shape().elem_count() / input_dim;
        let dev = x.device();
        let x = x.as_cuda_slice::<f32>()?.slice(x_start..x_end);
        let weight = weight
            .as_cuda_slice::<f32>()?
            .slice(weight_start..weight_end);
        let bias = bias.as_cuda_slice::<f32>()?.slice(bias_start..bias_end);
        // SAFETY: The validated matrix dimensions exactly cover the allocation.
        let mut output = unsafe { dev.alloc::<f32>(rows * output_dim)? };
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: *output_dim as u64,
            n: rows as u64,
            k: *input_dim as u64,
            alpha: 1.,
            lda: *input_dim as i64,
            ldb: *input_dim as i64,
            beta: 0.,
            ldc: *output_dim as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let host_profile = std::env::var_os("CANDLE_PROFILE_CUBLASLT_HOST").is_some();
        let host_start = host_profile.then(std::time::Instant::now);
        // SAFETY: All pointers, leading dimensions, and extents were validated above.
        unsafe {
            dev.cublas_lt()
                .matmul(cfg, &weight, &x, &mut output, Some(&bias), None)
        }
        .map_err(|err| candle::Error::Msg(format!("cuBLASLt linear failed: {err}")))?;
        if let Some(start) = host_start {
            eprintln!(
                "CANDLE_CUBLASLT_HOST rows={rows} input={input_dim} output={output_dim} us={:.3}",
                start.elapsed().as_secs_f64() * 1e6
            );
        }
        let mut output_dims = x_l.dims().to_vec();
        *output_dims.last_mut().unwrap() = *output_dim;
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: dev.clone(),
            },
            Shape::from_dims(&output_dims),
        ))
    }
}

#[cfg(feature = "cuda")]
fn linear_cuda_f32_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Option<Tensor>> {
    if !x.device().is_cuda()
        || x.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || x.dims().is_empty()
        || weight.dims().len() != 2
        || bias.dims().len() != 1
        || !x.is_contiguous()
        || !weight.is_contiguous()
        || !bias.is_contiguous()
    {
        return Ok(None);
    }
    Ok(Some(x.apply_op3_no_bwd(
        weight,
        bias,
        &LinearCudaF32Bias,
    )?))
}

/// CPU f32 contiguous `x @ weight.t() + bias` without an intermediate output.
///
/// Seed the final row-major destination with the bias, then ask GEMM to read
/// and accumulate into it. This matches `addmm`'s data flow and eliminates the
/// generic matmul result plus the following broadcast-add allocation.
fn linear_cpu_f32_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Option<Tensor>> {
    linear_cpu_f32_bias_impl(x, weight, bias, None, false, false, false, false)
}

fn linear_cpu_f32_bias_residual(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
) -> Result<Option<Tensor>> {
    linear_cpu_f32_bias_impl(x, weight, bias, Some(residual), false, false, false, false)
}

fn linear_cpu_f32_bias_impl(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    residual: Option<&Tensor>,
    gelu_erf: bool,
    relu: bool,
    force_short_row_split: bool,
    weight_is_transposed: bool,
) -> Result<Option<Tensor>> {
    use gemm::{gemm, Parallelism};

    configure_gemm_from_env();

    if !x.device().is_cpu()
        || x.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || x.dims().is_empty()
        || weight.dims().len() != 2
        || bias.dims().len() != 1
    {
        return Ok(None);
    }
    let Some((x_start, x_end)) = x.layout().contiguous_offsets() else {
        return Ok(None);
    };
    let Some((weight_start, weight_end)) = weight.layout().contiguous_offsets() else {
        return Ok(None);
    };
    let Some((bias_start, bias_end)) = bias.layout().contiguous_offsets() else {
        return Ok(None);
    };
    let input_dim = x.dims()[x.dims().len() - 1];
    let (weight_input_dim, output_dim) = if weight_is_transposed {
        (weight.dims()[0], weight.dims()[1])
    } else {
        (weight.dims()[1], weight.dims()[0])
    };
    if weight_input_dim != input_dim
        || bias.dims()[0] != output_dim
        || input_dim == 0
        || output_dim == 0
    {
        return Ok(None);
    }
    let output_dims = {
        let mut dims = x.dims().to_vec();
        *dims.last_mut().expect("non-empty dimensions checked above") = output_dim;
        dims
    };
    let (x_storage, _) = x.storage_and_layout();
    let (weight_storage, _) = weight.storage_and_layout();
    let (bias_storage, _) = bias.storage_and_layout();
    let (Storage::Cpu(x_storage), Storage::Cpu(weight_storage), Storage::Cpu(bias_storage)) =
        (&*x_storage, &*weight_storage, &*bias_storage)
    else {
        return Ok(None);
    };
    let x_data = &x_storage.as_slice::<f32>()?[x_start..x_end];
    let weight_data = &weight_storage.as_slice::<f32>()?[weight_start..weight_end];
    let bias_data = &bias_storage.as_slice::<f32>()?[bias_start..bias_end];
    let rows = x_data.len() / input_dim;
    if rows * input_dim != x_data.len()
        || weight_data.len() != output_dim * input_dim
        || bias_data.len() != output_dim
    {
        return Ok(None);
    }

    if let Some(residual) = residual {
        if !residual.device().is_cpu()
            || residual.dtype() != DType::F32
            || residual.dims() != output_dims
        {
            return Ok(None);
        }
        let Some((start, end)) = residual.layout().contiguous_offsets() else {
            return Ok(None);
        };
        if end - start != rows * output_dim {
            return Ok(None);
        }
    }
    let residual_storage = residual.map(Tensor::storage_and_layout);
    let residual_data = if let Some((storage, layout)) = residual_storage.as_ref() {
        let Storage::Cpu(storage) = &**storage else {
            return Ok(None);
        };
        let (start, end) = layout.contiguous_offsets().expect("validated above");
        Some(&storage.as_slice::<f32>()?[start..end])
    } else {
        None
    };
    let output_len = rows * output_dim;
    let short_row_split = (force_short_row_split
        || (rayon::current_num_threads() >= 4
            && std::env::var_os("CANDLE_CPU_LINEAR_SHORTROW_SPLIT")
                .is_none_or(|value| value != "0")))
        && rows >= 2
        && rows <= 16
        && input_dim >= 1024
        && output_dim >= 128
        && !weight_is_transposed
        && residual.is_none()
        && !gelu_erf
        && !relu;
    let mut output;
    if short_row_split {
        // These small outputs do not repay a parallel bias-seeding pass.
        output = vec![0f32; output_len];
        for row in output.chunks_mut(output_dim) {
            row.copy_from_slice(bias_data);
        }
    } else if std::env::var_os("CANDLE_CPU_LINEAR_SINGLE_WRITE_SEED")
        .is_none_or(|value| value != "0")
    {
        output = Vec::with_capacity(output_len);
        let seed_row = |row_index: usize, row: &mut [std::mem::MaybeUninit<f32>]| {
            if let Some(data) = residual_data {
                let residual = &data[row_index * output_dim..(row_index + 1) * output_dim];
                for ((output, residual), bias) in row.iter_mut().zip(residual).zip(bias_data) {
                    output.write(residual + bias);
                }
            } else {
                for (output, bias) in row.iter_mut().zip(bias_data) {
                    output.write(*bias);
                }
            }
        };
        // Spawning Rayon work for the decoder's tiny token projections costs
        // more than copying their 1--4 KiB bias-seeded outputs. Keep those
        // writes local while preserving parallel seeding for large encoder
        // and AMG matrices. No input, weight, or per-worker buffer is copied.
        let parallel_seed_min = *LINEAR_PARALLEL_SEED_MIN.get_or_init(|| {
            std::env::var_os("CANDLE_CPU_LINEAR_PARALLEL_SEED_MIN")
                .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
                .unwrap_or(16 * 1024)
        });
        if output_len < parallel_seed_min || rayon::current_num_threads() == 1 {
            for (row_index, row) in output.spare_capacity_mut()[..output_len]
                .chunks_mut(output_dim)
                .enumerate()
            {
                seed_row(row_index, row);
            }
        } else {
            output.spare_capacity_mut()[..output_len]
                .par_chunks_mut(output_dim)
                .enumerate()
                .for_each(|(row_index, row)| seed_row(row_index, row));
        }
        // SAFETY: every element in the spare capacity is written exactly once
        // by a disjoint Rayon chunk before the vector length is increased.
        unsafe { output.set_len(output_len) };
    } else {
        output = vec![0f32; output_len];
        output
            .par_chunks_mut(output_dim)
            .enumerate()
            .for_each(|(row_index, row)| {
                if let Some(data) = residual_data {
                    let residual = &data[row_index * output_dim..(row_index + 1) * output_dim];
                    for ((output, residual), bias) in row.iter_mut().zip(residual).zip(bias_data) {
                        *output = residual + bias;
                    }
                } else {
                    row.copy_from_slice(bias_data);
                }
            });
    }
    let parallelism = if rayon::current_num_threads() > 1 {
        Parallelism::Rayon(rayon::current_num_threads())
    } else {
        Parallelism::None
    };
    let transpose_gemm = *TRANSPOSE_GEMM.get_or_init(|| {
        std::env::var_os("CANDLE_CPU_LINEAR_TRANSPOSE_GEMM").is_some_and(|value| value == "1")
    });
    if short_row_split {
        let chunk = std::env::var_os("CANDLE_CPU_LINEAR_SHORTROW_CHUNK")
            .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
            .filter(|&value| value > 0)
            .unwrap_or(if output_dim <= 512 { 16 } else { 32 });
        let output_address = output.as_mut_ptr() as usize;
        (0..output_dim.div_ceil(chunk))
            .into_par_iter()
            .for_each(|tile| {
                let first = tile * chunk;
                let width = (output_dim - first).min(chunk);
                // SAFETY: each tile owns a disjoint output-column interval in
                // every row. Input and weights are shared read-only; the GEMM
                // call completes before `output` is used again.
                unsafe {
                    gemm(
                        rows,
                        width,
                        input_dim,
                        (output_address as *mut f32).add(first),
                        1,
                        output_dim as isize,
                        true,
                        x_data.as_ptr(),
                        1,
                        input_dim as isize,
                        weight_data.as_ptr().add(first * input_dim),
                        input_dim as isize,
                        1,
                        1.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
            });
    } else {
        unsafe {
            // SAFETY: all slices have been validated contiguous with the exact
            // matrix extents above. `read_dst` is true because each row starts as
            // a complete copy of `bias_data`. The optional orientation computes
            // (X Wᵀ)ᵀ = W Xᵀ into the same row-major output without a copy.
            if weight_is_transposed {
                gemm(
                    rows,
                    output_dim,
                    input_dim,
                    output.as_mut_ptr(),
                    1,
                    output_dim as isize,
                    true,
                    x_data.as_ptr(),
                    1,
                    input_dim as isize,
                    weight_data.as_ptr(),
                    1,
                    output_dim as isize,
                    1.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            } else if transpose_gemm {
                gemm(
                    output_dim,
                    rows,
                    input_dim,
                    output.as_mut_ptr(),
                    output_dim as isize,
                    1,
                    true,
                    weight_data.as_ptr(),
                    1,
                    input_dim as isize,
                    x_data.as_ptr(),
                    input_dim as isize,
                    1,
                    1.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            } else {
                gemm(
                    rows,
                    output_dim,
                    input_dim,
                    output.as_mut_ptr(),
                    1,
                    output_dim as isize,
                    true,
                    x_data.as_ptr(),
                    1,
                    input_dim as isize,
                    weight_data.as_ptr(),
                    input_dim as isize,
                    1,
                    1.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    parallelism,
                );
            }
        }
    }
    if gelu_erf {
        candle::op::gelu_erf_f32_inplace(&mut output);
    } else if relu {
        let apply = |value: &mut f32| *value = value.max(0f32);
        if output.len() >= 16 * 1024 && rayon::current_num_threads() > 1 {
            output.par_iter_mut().for_each(apply);
        } else {
            output.iter_mut().for_each(apply);
        }
    }
    Ok(Some(Tensor::from_vec(output, output_dims, x.device())?))
}

#[derive(Debug)]
struct BiasAddLastDim;

impl candle::CustomOp2 for BiasAddLastDim {
    fn name(&self) -> &'static str {
        "bias-add-last-dim"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("bias-add-last-dim fast path is only implemented for cuda")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::{kernels, CudaStorage, CudaStorageSlice, WrapErr};

        if s1.dtype() != candle::DType::F32 || s2.dtype() != candle::DType::F32 {
            candle::bail!("bias-add-last-dim only supports f32")
        }
        let Some((o1, o2)) = l1.contiguous_offsets() else {
            candle::bail!("bias-add-last-dim input must be contiguous")
        };
        let Some((b1, b2)) = l2.contiguous_offsets() else {
            candle::bail!("bias-add-last-dim bias must be contiguous")
        };
        let dims = l1.dims();
        let Some(&n_cols) = dims.last() else {
            candle::bail!("bias-add-last-dim input must have rank greater than zero")
        };
        if l2.shape().elem_count() != n_cols {
            candle::bail!("bias-add-last-dim bias length mismatch")
        }
        let src = s1.as_cuda_slice::<f32>()?.slice(o1..o2);
        let bias = s2.as_cuda_slice::<f32>()?.slice(b1..b2);
        let dev = s1.device();
        let elem_count = l1.shape().elem_count();
        // SAFETY: Set by running the kernel below.
        let dst = unsafe { dev.alloc::<f32>(elem_count)? };
        let func = dev.get_or_load_func("bias_add_last_dim_f32", &kernels::BINARY)?;
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let mut builder = func.builder();
        builder.arg(&src);
        builder.arg(&bias);
        builder.arg(&dst);
        candle::builder_arg!(builder, elem_count);
        candle::builder_arg!(builder, n_cols);
        // SAFETY: ffi.
        unsafe { builder.launch(cfg) }.w()?;
        let storage = CudaStorage {
            slice: CudaStorageSlice::F32(dst),
            device: dev.clone(),
        };
        Ok((storage, l1.shape().clone()))
    }
}

fn can_use_cuda_bias_add_last_dim(xs: &Tensor, bias: &Tensor) -> bool {
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let (bias_storage, bias_layout) = bias.storage_and_layout();
    matches!(
        (&*xs_storage, &*bias_storage),
        (candle::Storage::Cuda(_), candle::Storage::Cuda(_))
    ) && xs_storage.dtype() == candle::DType::F32
        && bias_storage.dtype() == candle::DType::F32
        && xs_layout.contiguous_offsets().is_some()
        && bias_layout.contiguous_offsets().is_some()
        && xs_layout
            .dims()
            .last()
            .is_some_and(|&n_cols| bias_layout.shape().elem_count() == n_cols)
}

fn bias_add_last_dim(xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if can_use_cuda_bias_add_last_dim(xs, bias) {
        xs.apply_op2_no_bwd(bias, &BiasAddLastDim)
    } else {
        xs.broadcast_add(bias)
    }
}

/// Create or initialize a new linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(Linear::new(ws, Some(bs)))
}

/// Create or initialize a new linear layer without biases.
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<Linear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(Linear::new(ws, None))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb)
    } else {
        linear_no_bias(in_dim, out_dim, vb)
    }
}

#[cfg(test)]
mod tests {
    use super::{bias_add_last_dim, linear_cpu_f32_bias, linear_cpu_f32_bias_impl, Linear};
    use crate::Module;
    use candle::{Device, Result, Tensor};

    fn assert_close(lhs: &Tensor, rhs: &Tensor, threshold: f32) -> Result<()> {
        let max_diff = (lhs - rhs)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .reshape(())?
            .to_vec0::<f32>()?;
        assert!(max_diff < threshold, "max_diff {max_diff}");
        Ok(())
    }

    fn bias_add_last_dim_matches_broadcast_add_3d(device: &Device) -> Result<()> {
        let xs = Tensor::from_vec(
            (0..2 * 4 * 6)
                .map(|i| (i as f32 % 17.) * 0.125 - 0.75)
                .collect(),
            (2, 4, 6),
            device,
        )?;
        let bias = Tensor::from_vec((0..6).map(|i| i as f32 * 0.2 - 0.5).collect(), 6, device)?;
        let actual = bias_add_last_dim(&xs, &bias)?;
        let expected = xs.broadcast_add(&bias)?;
        assert_close(&actual, &expected, 1e-6)
    }

    fn linear_bias_add_matches_broadcast_add_2d(device: &Device) -> Result<()> {
        let xs = Tensor::from_vec(
            (0..4 * 5).map(|i| (i as f32 % 13.) * 0.2 - 0.6).collect(),
            (4, 5),
            device,
        )?;
        let weight = Tensor::from_vec(
            (0..3 * 5).map(|i| (i as f32 % 7.) * 0.125 - 0.25).collect(),
            (3, 5),
            device,
        )?;
        let bias = Tensor::new(&[-0.5f32, 0.25, 0.75], device)?;
        let layer = Linear::new(weight.clone(), Some(bias.clone()));
        let actual = layer.forward(&xs)?;
        let expected = xs.matmul(&weight.t()?)?.broadcast_add(&bias)?;
        assert_close(&actual, &expected, 1e-6)
    }

    fn linear_bias_add_matches_broadcast_add_4d(device: &Device) -> Result<()> {
        let xs = Tensor::from_vec(
            (0..2 * 3 * 4 * 5)
                .map(|i| (i as f32 % 19.) * 0.1 - 0.9)
                .collect(),
            (2, 3, 4, 5),
            device,
        )?;
        let weight = Tensor::from_vec(
            (0..6 * 5).map(|i| (i as f32 % 11.) * 0.15 - 0.45).collect(),
            (6, 5),
            device,
        )?;
        let bias = Tensor::from_vec((0..6).map(|i| i as f32 * 0.2 - 0.5).collect(), 6, device)?;
        let layer = Linear::new(weight.clone(), Some(bias.clone()));
        let actual = layer.forward(&xs)?;
        let expected = xs
            .reshape((2 * 3 * 4, 5))?
            .matmul(&weight.t()?)?
            .reshape((2, 3, 4, ()))?
            .broadcast_add(&bias)?;
        assert_close(&actual, &expected, 1e-5)
    }

    #[test]
    fn linear_add_residual_matches_composition_cpu() -> Result<()> {
        let device = &Device::Cpu;
        let xs = Tensor::from_vec(
            (0..2 * 3 * 5)
                .map(|i| (i as f32 % 13.) * 0.15 - 0.6)
                .collect(),
            (2, 3, 5),
            device,
        )?;
        let weight = Tensor::from_vec(
            (0..4 * 5).map(|i| (i as f32 % 7.) * 0.2 - 0.4).collect(),
            (4, 5),
            device,
        )?;
        let bias = Tensor::from_vec((0..4).map(|i| i as f32 * 0.1 - 0.15).collect(), 4, device)?;
        let residual = Tensor::from_vec(
            (0..2 * 3 * 4)
                .map(|i| (i as f32 % 11.) * 0.1 - 0.3)
                .collect(),
            (2, 3, 4),
            device,
        )?;
        let layer = Linear::new(weight, Some(bias));
        let actual = layer.forward_add_residual(&xs, &residual)?;
        let expected = residual.broadcast_add(&layer.forward(&xs)?)?;
        assert_close(&actual, &expected, 1e-5)
    }

    #[test]
    fn bias_add_last_dim_matches_broadcast_add_3d_cpu() -> Result<()> {
        bias_add_last_dim_matches_broadcast_add_3d(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn bias_add_last_dim_matches_broadcast_add_3d_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        bias_add_last_dim_matches_broadcast_add_3d(&device)
    }

    #[test]
    fn linear_bias_add_matches_broadcast_add_2d_cpu() -> Result<()> {
        linear_bias_add_matches_broadcast_add_2d(&Device::Cpu)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn linear_bias_add_matches_broadcast_add_2d_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        linear_bias_add_matches_broadcast_add_2d(&device)
    }

    #[test]
    fn linear_bias_add_matches_broadcast_add_4d_cpu() -> Result<()> {
        linear_bias_add_matches_broadcast_add_4d(&Device::Cpu)
    }

    #[test]
    fn fused_cpu_linear_bias_matches_matmul_then_broadcast_add() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (2, 3, 5), &device)?;
        let weight = Tensor::randn(0f32, 1f32, (7, 5), &device)?;
        let bias = Tensor::randn(0f32, 1f32, 7, &device)?;
        let fused = linear_cpu_f32_bias(&x, &weight, &bias)?.expect("eligible f32 CPU tensors");
        let expected = x
            .reshape((6, 5))?
            .matmul(&weight.t()?)?
            .reshape((2, 3, 7))?
            .broadcast_add(&bias)?;
        assert_close(&fused, &expected, 1e-5)
    }

    #[test]
    fn prepared_transposed_cpu_linear_matches_fused_linear() -> Result<()> {
        let device = Device::Cpu;
        for rows in [8, 4096] {
            let x = Tensor::from_vec(
                (0..rows * 32)
                    .map(|index| (index as f32 * 0.013).sin() * 0.125)
                    .collect(),
                (1, rows, 32),
                &device,
            )?;
            let weight = Tensor::from_vec(
                (0..48 * 32)
                    .map(|index| (index as f32 * 0.007).cos() * 0.03125)
                    .collect(),
                (48, 32),
                &device,
            )?;
            let bias = Tensor::from_vec(
                (0..48).map(|index| index as f32 * 0.001 - 0.05).collect(),
                48,
                &device,
            )?;
            let layer = Linear::new(weight, Some(bias));
            let prepared = layer
                .prepare_transposed_weight_cpu_f32()?
                .expect("eligible CPU f32 weight");
            let actual = layer.forward_fused_bias_cpu_pretransposed(&x, &prepared)?;
            let expected = layer.forward_fused_bias_cpu(&x)?;
            assert_close(&actual, &expected, 1e-4)?;
        }
        Ok(())
    }

    #[test]
    fn short_row_split_matches_matmul_with_channel_tail() -> Result<()> {
        let device = Device::Cpu;
        for (rows, input_dim, output_dim) in [(2, 1024, 130), (8, 2048, 256)] {
            let x = Tensor::from_vec(
                (0..rows * input_dim)
                    .map(|index| (index as f32 * 0.013).sin() * 0.125)
                    .collect(),
                (rows, input_dim),
                &device,
            )?;
            let weight = Tensor::from_vec(
                (0..output_dim * input_dim)
                    .map(|index| (index as f32 * 0.007).cos() * 0.03125)
                    .collect(),
                (output_dim, input_dim),
                &device,
            )?;
            let bias = Tensor::from_vec(
                (0..output_dim)
                    .map(|index| index as f32 * 0.001 - 0.05)
                    .collect(),
                output_dim,
                &device,
            )?;
            let split =
                linear_cpu_f32_bias_impl(&x, &weight, &bias, None, false, false, true, false)?
                    .expect("eligible short-row CPU f32 linear");
            let expected = x.matmul(&weight.t()?)?.broadcast_add(&bias)?;
            assert_close(&split, &expected, 1e-4)?;
        }
        Ok(())
    }

    #[test]
    fn fused_cpu_linear_gelu_matches_separate_exact_gelu() -> Result<()> {
        let device = Device::Cpu;
        // Cross the 16K element parallel/vector threshold used by exact GELU.
        let x = Tensor::from_vec(
            (0..256 * 64)
                .map(|i| (i as f32 % 37.) * 0.025 - 0.45)
                .collect(),
            (256, 64),
            &device,
        )?;
        let weight = Tensor::from_vec(
            (0..128 * 64)
                .map(|i| (i as f32 % 29.) * 0.01 - 0.14)
                .collect(),
            (128, 64),
            &device,
        )?;
        let bias = Tensor::from_vec(
            (0..128).map(|i| (i as f32 % 17.) * 0.03 - 0.2).collect(),
            128,
            &device,
        )?;
        let layer = Linear::new(weight, Some(bias));
        let fused = layer.forward_fused_bias_gelu_erf_cpu(&x)?;
        let separate = layer.forward_fused_bias_cpu(&x)?.gelu_erf()?;
        assert_close(&fused, &separate, 1e-6)
    }

    #[test]
    fn fused_cpu_linear_relu_matches_separate_relu() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..256 * 64)
                .map(|i| (i as f32 % 37.) * 0.025 - 0.45)
                .collect(),
            (256, 64),
            &device,
        )?;
        let weight = Tensor::from_vec(
            (0..128 * 64)
                .map(|i| (i as f32 % 29.) * 0.01 - 0.14)
                .collect(),
            (128, 64),
            &device,
        )?;
        let bias = Tensor::from_vec(
            (0..128).map(|i| (i as f32 % 17.) * 0.03 - 0.2).collect(),
            128,
            &device,
        )?;
        let layer = Linear::new(weight, Some(bias));
        let fused = layer.forward_fused_bias_relu_cpu(&x)?;
        let separate = layer.forward_fused_bias_cpu(&x)?.relu()?;
        assert_close(&fused, &separate, 1e-6)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn linear_bias_add_matches_broadcast_add_4d_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        linear_bias_add_matches_broadcast_add_4d(&device)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn linear_materializes_non_contiguous_4d_cuda_input() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let xs = Tensor::arange(0f32, (2 * 3 * 4 * 5) as f32, &device)?
            .reshape((2, 5, 3, 4))?
            .permute((0, 2, 3, 1))?;
        assert!(!xs.is_contiguous());
        let weight = Tensor::from_vec(
            (0..6 * 5).map(|index| index as f32 * 0.01).collect(),
            (6, 5),
            &device,
        )?;
        let layer = Linear::new(weight.clone(), None);
        let actual = layer.forward(&xs)?;
        let expected = xs
            .contiguous()?
            .reshape((2 * 3 * 4, 5))?
            .matmul(&weight.t()?)?
            .reshape((2, 3, 4, 6))?;
        assert_close(&actual, &expected, 1e-5)
    }
}
