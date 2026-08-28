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
use candle::{CpuStorage, Layout, Result, Shape, Tensor};

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
}

impl super::Module for Linear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
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
    use super::{bias_add_last_dim, Linear};
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

    #[cfg(feature = "cuda")]
    #[test]
    fn linear_bias_add_matches_broadcast_add_4d_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        linear_bias_add_matches_broadcast_add_4d(&device)
    }
}
