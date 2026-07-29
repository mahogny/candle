use candle::{DType, IndexOp, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct ImageEncoderViTTiming {
    pub patch_embed_s: f64,
    pub pos_embed_s: f64,
    pub blocks_s: f64,
    pub neck_s: f64,
    pub total_s: f64,
    pub blocks: Vec<BlockTiming>,
}

#[derive(Clone, Debug, Default)]
pub struct BlockTiming {
    pub norm1_s: f64,
    pub partition_s: f64,
    pub attention_s: f64,
    pub unpartition_s: f64,
    pub residual_s: f64,
    pub norm2_s: f64,
    pub mlp_lin1_s: f64,
    pub mlp_lin1_matmul_s: f64,
    pub mlp_lin1_bias_s: f64,
    pub mlp_activation_s: f64,
    pub mlp_lin2_s: f64,
    pub mlp_lin2_matmul_s: f64,
    pub mlp_lin2_bias_s: f64,
    pub mlp_residual_s: f64,
    pub norm2_mlp_s: f64,
    pub total_s: f64,
    pub attention: AttentionTiming,
}

#[derive(Clone, Debug, Default)]
pub struct AttentionTiming {
    pub qkv_s: f64,
    pub qkv_matmul_s: f64,
    pub qkv_bias_s: f64,
    pub qk_matmul_s: f64,
    pub rel_pos_s: f64,
    pub softmax_s: f64,
    pub av_matmul_s: f64,
    pub proj_s: f64,
    pub proj_matmul_s: f64,
    pub proj_bias_s: f64,
    pub total_s: f64,
}

fn sync_for_timing(xs: &Tensor) -> Result<()> {
    xs.device().synchronize()
}

fn timed_linear(
    linear: &super::Linear,
    xs: &Tensor,
    total_s: &mut f64,
    matmul_s: &mut f64,
    bias_s: &mut f64,
) -> Result<Tensor> {
    let start = Instant::now();
    let ys = linear.forward(xs)?;
    sync_for_timing(&ys)?;
    *total_s += start.elapsed().as_secs_f64();
    *matmul_s = 0.0;
    *bias_s = 0.0;
    Ok(ys)
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    span: tracing::Span,
}

impl PatchEmbed {
    fn new(
        in_chans: usize,
        embed_dim: usize,
        k_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, k_size, cfg, vb.pp("proj"))?;
        let span = tracing::span!(tracing::Level::TRACE, "patch-embed");
        Ok(Self { proj, span })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        xs.apply(&self.proj)?.permute((0, 2, 3, 1))
    }
}

// A custom op to make add_decomposed_rel_pos faster. Most of the time is spent on the final
// addition in the case where b = 12, q_h = q_w = 4096, k_h = k_w = 4096
//   (attn.reshape((b, q_h, q_w, k_h, k_w))?
//       + rel_h.unsqueeze(4)?.broadcast_add(&rel_w.unsqueeze(3)?)?)?
//   .reshape((b, q_h * q_w, k_h * k_w))
// Ideally we would perform this operation in place but this is not supported in candle at the
// moment. We should also investigate using f16 rather than f32.
struct Add3(usize, usize, usize, usize, usize);
impl candle::CustomOp3 for Add3 {
    fn name(&self) -> &'static str {
        "add3"
    }

    fn cpu_fwd(
        &self,
        s1: &candle::CpuStorage,
        l1: &candle::Layout,
        s2: &candle::CpuStorage,
        l2: &candle::Layout,
        s3: &candle::CpuStorage,
        l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        use rayon::prelude::*;

        let Add3(b, q_h, q_w, k_h, k_w) = *self;
        let s1 = s1.as_slice::<f32>()?;
        let s1 = match l1.contiguous_offsets() {
            None => candle::bail!("input1 has to be contiguous"),
            Some((o1, o2)) => &s1[o1..o2],
        };
        let s2 = s2.as_slice::<f32>()?;
        let s2 = match l2.contiguous_offsets() {
            None => candle::bail!("input2 has to be contiguous"),
            Some((o1, o2)) => &s2[o1..o2],
        };
        let s3 = s3.as_slice::<f32>()?;
        let s3 = match l3.contiguous_offsets() {
            None => candle::bail!("input3 has to be contiguous"),
            Some((o1, o2)) => &s3[o1..o2],
        };
        let mut dst = vec![0f32; b * q_h * q_w * k_h * k_w];
        dst.par_chunks_exact_mut(k_h * k_w)
            .enumerate()
            .for_each(|(b_idx, dst)| {
                let s1_idx = b_idx * k_h * k_w;
                let s2_idx = b_idx * k_h;
                let s3_idx = b_idx * k_w;
                for h_idx in 0..k_h {
                    let s1_idx = s1_idx + h_idx * k_w;
                    let s2_idx = s2_idx + h_idx;
                    let dst_idx = h_idx * k_w;
                    for w_idx in 0..k_w {
                        let s1_idx = s1_idx + w_idx;
                        let s3_idx = s3_idx + w_idx;
                        let dst_idx = dst_idx + w_idx;
                        dst[dst_idx] = s1[s1_idx] + s2[s2_idx] + s3[s3_idx]
                    }
                }
            });
        let dst = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((dst, (b, q_h * q_w, k_h * k_w).into()))
    }
}

#[cfg(feature = "cuda")]
struct Add3Inplace(usize, usize, usize, usize, usize);

#[cfg(feature = "cuda")]
impl candle::InplaceOp3 for Add3Inplace {
    fn name(&self) -> &'static str {
        "add3-inplace"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<()> {
        candle::bail!("add3-inplace is only implemented for cuda")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &mut candle::CudaStorage,
        l1: &candle::Layout,
        s2: &candle::CudaStorage,
        l2: &candle::Layout,
        s3: &candle::CudaStorage,
        l3: &candle::Layout,
    ) -> Result<()> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::cudarc::nvrtc;
        use candle::cuda_backend::{CudaStorageSlice, WrapErr};

        const MODULE_NAME: &str = "sam_add3_inplace_f32";
        const FUNC_NAME: &str = "sam_add3_inplace_f32";
        const CUDA_SRC: &str = r#"
extern "C" __global__ void sam_add3_inplace_f32(
    float* __restrict__ attn,
    const float* __restrict__ rel_h,
    const float* __restrict__ rel_w,
    unsigned long long elem_count,
    unsigned int kh,
    unsigned int kw) {
    unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= elem_count) {
        return;
    }

    unsigned int w = idx % kw;
    unsigned int h = (idx / kw) % kh;
    unsigned long long q_idx = idx / ((unsigned long long)kh * kw);
    attn[idx] += rel_h[q_idx * kh + h] + rel_w[q_idx * kw + w];
}
"#;

        let Add3Inplace(b, q_h, q_w, k_h, k_w) = *self;
        let expected_attn = b * q_h * q_w * k_h * k_w;
        let expected_rel_h = b * q_h * q_w * k_h;
        let expected_rel_w = b * q_h * q_w * k_w;
        if l1.shape().elem_count() != expected_attn
            || l2.shape().elem_count() != expected_rel_h
            || l3.shape().elem_count() != expected_rel_w
        {
            candle::bail!(
                "unexpected add3 shapes {:?} {:?} {:?} for dims ({b}, {q_h}, {q_w}, {k_h}, {k_w})",
                l1.shape(),
                l2.shape(),
                l3.shape()
            )
        }

        let attn = s1.as_cuda_slice::<f32>()?;
        let attn = match l1.contiguous_offsets() {
            None => candle::bail!("attn input has to be contiguous"),
            Some((o1, o2)) => attn.slice(o1..o2),
        };
        let (rel_h, rel_w) = match (&s2.slice, &s3.slice) {
            (CudaStorageSlice::F32(rel_h), CudaStorageSlice::F32(rel_w)) => (rel_h, rel_w),
            _ => candle::bail!(
                "unsupported dtype for cuda add3-inplace {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
        };
        let rel_h = match l2.contiguous_offsets() {
            None => candle::bail!("rel_h input has to be contiguous"),
            Some((o1, o2)) => rel_h.slice(o1..o2),
        };
        let rel_w = match l3.contiguous_offsets() {
            None => candle::bail!("rel_w input has to be contiguous"),
            Some((o1, o2)) => rel_w.slice(o1..o2),
        };

        static PTX: std::sync::OnceLock<String> = std::sync::OnceLock::new();

        let dev = s1.device();
        let ptx = match PTX.get() {
            Some(ptx) => ptx,
            None => {
                let ptx = nvrtc::compile_ptx(CUDA_SRC).w()?.to_src();
                let _ = PTX.set(ptx);
                PTX.get()
                    .expect("PTX cache must be initialized after successful compilation")
            }
        };
        let func = dev.get_or_load_custom_func(FUNC_NAME, MODULE_NAME, ptx)?;
        let cfg = LaunchConfig::for_num_elems(expected_attn as u32);
        let mut builder = func.builder();
        builder.arg(&attn);
        builder.arg(&rel_h);
        builder.arg(&rel_w);
        candle::builder_arg!(builder, expected_attn as u64, k_h as u32, k_w as u32);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

#[derive(Debug)]
struct Attention {
    qkv: super::Linear,
    proj: super::Linear,
    num_heads: usize,
    scale: f64,
    rel_pos_hw: Option<(Tensor, Tensor)>,
    span: tracing::Span,
    span_matmul: tracing::Span,
    span_rel_pos: tracing::Span,
    span_softmax: tracing::Span,
}

impl Attention {
    fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        input_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attention");
        let span_matmul = tracing::span!(tracing::Level::TRACE, "attn-matmul");
        let span_rel_pos = tracing::span!(tracing::Level::TRACE, "attn-rel-pos");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "attn-sm");
        let qkv = super::linear(vb.pp("qkv"), dim, dim * 3, qkv_bias)?;
        let proj = super::linear(vb.pp("proj"), dim, dim, true)?;
        let head_dim = dim / num_heads;
        let scale = 1. / (head_dim as f64).sqrt();
        let rel_pos_hw = if use_rel_pos {
            let h = vb.get((2 * input_size.0 - 1, head_dim), "rel_pos_h")?;
            let w = vb.get((2 * input_size.1 - 1, head_dim), "rel_pos_w")?;
            Some((h, w))
        } else {
            None
        };
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            rel_pos_hw,
            span,
            span_matmul,
            span_rel_pos,
            span_softmax,
        })
    }

    fn add_decomposed_rel_pos(
        &self,
        attn: Tensor,
        q: &Tensor,
        (q_h, q_w): (usize, usize),
        (k_h, k_w): (usize, usize),
    ) -> Result<Tensor> {
        match &self.rel_pos_hw {
            Some((rel_pos_h, rel_pos_w)) => {
                let r_h = get_rel_pos(q_h, k_h, rel_pos_h)?;
                let r_w = get_rel_pos(q_w, k_w, rel_pos_w)?;
                let (b, _, dim) = q.dims3()?;
                let r_q = q.reshape((b, q_h, q_w, dim))?;
                // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
                let rel_h = r_q.matmul(&r_h.broadcast_left(b)?.t()?.contiguous()?)?;
                // rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
                let rel_w = r_q
                    .transpose(1, 2)? // -> bwhc
                    .contiguous()?
                    .matmul(&r_w.broadcast_left(b)?.t()?.contiguous()?)? // bwhc,bwck -> bwhk
                    .transpose(1, 2)?
                    .contiguous()?;
                if attn.device().is_cpu() {
                    let op = Add3(b, q_h, q_w, k_h, k_w);
                    attn.apply_op3_no_bwd(&rel_h, &rel_w, &op)
                } else {
                    #[cfg(feature = "cuda")]
                    if attn.device().is_cuda() {
                        let attn = attn.reshape((b, q_h, q_w, k_h, k_w))?;
                        let op = Add3Inplace(b, q_h, q_w, k_h, k_w);
                        attn.inplace_op3(&rel_h, &rel_w, &op)?;
                        return attn.reshape((b, q_h * q_w, k_h * k_w));
                    }
                    let attn = attn.reshape((b, q_h, q_w, k_h, k_w))?;
                    (attn + rel_h.unsqueeze(4)?.broadcast_add(&rel_w.unsqueeze(3)?)?)?.reshape((
                        b,
                        q_h * q_w,
                        k_h * k_w,
                    ))
                }
            }
            None => Ok(attn),
        }
    }
}

fn get_rel_pos(q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
    let max_rel_dist = 2 * usize::max(q_size, k_size) - 1;
    let dev = rel_pos.device();
    let rel_pos_resized = if rel_pos.dim(0)? != max_rel_dist {
        todo!("interpolation")
    } else {
        rel_pos
    };
    let q_coords = Tensor::arange(0u32, q_size as u32, dev)?
        .reshape((q_size, 1))?
        .to_dtype(DType::F32)?;
    let k_coords = Tensor::arange(0u32, k_size as u32, dev)?
        .reshape((1, k_size))?
        .to_dtype(DType::F32)?;
    let q_coords = (q_coords * f64::max(1f64, k_size as f64 / q_size as f64))?;
    let k_coords = (k_coords * f64::max(1f64, q_size as f64 / k_size as f64))?;
    let relative_coords = (q_coords.broadcast_sub(&k_coords)?
        + (k_size as f64 - 1.) * f64::max(1f64, q_size as f64 / k_size as f64))?;
    let (d1, d2) = relative_coords.dims2()?;
    let relative_coords = relative_coords.to_dtype(DType::U32)?;
    rel_pos_resized
        .index_select(&relative_coords.reshape(d1 * d2)?, 0)?
        .reshape((d1, d2, ()))
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_impl(xs, None)
    }
}

impl Attention {
    fn forward_with_timing(&self, xs: &Tensor, timing: &mut AttentionTiming) -> Result<Tensor> {
        self.forward_impl(xs, Some(timing))
    }

    fn forward_impl(
        &self,
        xs: &Tensor,
        mut timing: Option<&mut AttentionTiming>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let total_start = Instant::now();
        let (b, h, w, c) = xs.dims4()?;
        let t0 = Instant::now();
        let qkv_in = xs.flatten_to(1)?;
        let qkv = if let Some(t) = timing.as_deref_mut() {
            timed_linear(
                &self.qkv,
                &qkv_in,
                &mut t.qkv_s,
                &mut t.qkv_matmul_s,
                &mut t.qkv_bias_s,
            )?
        } else {
            self.qkv.forward(&qkv_in)?
        }
        .reshape((b, h * w, 3, self.num_heads, c / self.num_heads))?
        .permute((2, 0, 3, 1, 4))?
        .reshape((3, b * self.num_heads, h * w, c / self.num_heads))?;
        if timing.is_some() {
            sync_for_timing(&qkv)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            if t.qkv_s == 0.0 {
                t.qkv_s += t0.elapsed().as_secs_f64();
            }
        }
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let t0 = Instant::now();
        let attn = {
            let _enter = self.span_matmul.enter();
            (&q * self.scale)?.matmul(&k.t()?)?
        };
        if timing.is_some() {
            sync_for_timing(&attn)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.qk_matmul_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let attn = {
            let _enter = self.span_rel_pos.enter();
            self.add_decomposed_rel_pos(attn, &q, (h, w), (h, w))?
        };
        if timing.is_some() {
            sync_for_timing(&attn)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.rel_pos_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let attn = {
            let _enter = self.span_softmax.enter();
            candle_nn::ops::softmax_last_dim(&attn)?
        };
        if timing.is_some() {
            sync_for_timing(&attn)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.softmax_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let attn = {
            let _enter = self.span_matmul.enter();
            attn.matmul(&v)?
        };
        if timing.is_some() {
            sync_for_timing(&attn)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.av_matmul_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let attn = attn
            .reshape((b, self.num_heads, h, w, c / self.num_heads))?
            .permute((0, 2, 3, 1, 4))?
            .reshape((b, h * w, c))?;
        let out = if let Some(t) = timing.as_deref_mut() {
            timed_linear(
                &self.proj,
                &attn,
                &mut t.proj_s,
                &mut t.proj_matmul_s,
                &mut t.proj_bias_s,
            )?
        } else {
            self.proj.forward(&attn)?
        }
        .reshape((b, h, w, c))?;
        if timing.is_some() {
            sync_for_timing(&out)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            if t.proj_s == 0.0 {
                t.proj_s += t0.elapsed().as_secs_f64();
            }
            t.total_s += total_start.elapsed().as_secs_f64();
        }
        Ok(out)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: super::MlpBlock,
    window_size: usize,
    span: tracing::Span,
}

impl Block {
    fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        window_size: usize,
        input_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-6, vb.pp("norm2"))?;
        let input_size_attn = if window_size == 0 {
            input_size
        } else {
            (window_size, window_size)
        };
        let attn = Attention::new(
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            input_size_attn,
            vb.pp("attn"),
        )?;
        let mlp = super::MlpBlock::new(dim, dim * 4, candle_nn::Activation::Gelu, vb.pp("mlp"))?;
        let span = tracing::span!(tracing::Level::TRACE, "ie-block");
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
            span,
        })
    }
}

fn window_partition(xs: Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = xs.dims4()?;
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;
    let xs = if pad_h > 0 {
        xs.pad_with_zeros(1, 0, pad_h)?
    } else {
        xs
    };
    let xs = if pad_w > 0 {
        xs.pad_with_zeros(2, 0, pad_w)?
    } else {
        xs
    };
    let (h_p, w_p) = (h + pad_h, w + pad_w);
    let windows = xs
        .reshape((
            b,
            h_p / window_size,
            window_size,
            w_p / window_size,
            window_size,
            c,
        ))?
        .transpose(2, 3)?
        .contiguous()?
        .flatten_to(2)?;
    Ok((windows, (h_p, w_p)))
}

fn window_unpartition(
    windows: Tensor,
    window_size: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
) -> Result<Tensor> {
    let b = windows.dim(0)? / (h_p * w_p / window_size / window_size);
    let xs = windows
        .reshape((
            b,
            h_p / window_size,
            w_p / window_size,
            window_size,
            window_size,
            windows.elem_count() / b / h_p / w_p,
        ))?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((b, h_p, w_p, ()))?;
    let xs = if h_p > h { xs.narrow(1, 0, h)? } else { xs };
    let xs = if w_p > w { xs.narrow(2, 0, w)? } else { xs };
    Ok(xs)
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_impl(xs, None)
    }
}

impl Block {
    fn forward_with_timing(&self, xs: &Tensor, timing: &mut BlockTiming) -> Result<Tensor> {
        self.forward_impl(xs, Some(timing))
    }

    fn forward_impl(&self, xs: &Tensor, mut timing: Option<&mut BlockTiming>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let total_start = Instant::now();
        let shortcut = xs;
        let t0 = Instant::now();
        let xs = self.norm1.forward(xs)?;
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.norm1_s += t0.elapsed().as_secs_f64();
        }
        let hw = (xs.dim(1)?, xs.dim(2)?);
        let t0 = Instant::now();
        let (xs, pad_hw) = if self.window_size > 0 {
            window_partition(xs, self.window_size)?
        } else {
            (xs, (0, 0))
        };
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.partition_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let xs = if let Some(t) = timing.as_deref_mut() {
            self.attn.forward_with_timing(&xs, &mut t.attention)?
        } else {
            self.attn.forward(&xs)?
        };
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.attention_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let xs = if self.window_size > 0 {
            window_unpartition(xs, self.window_size, pad_hw, hw)?
        } else {
            xs
        };
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.unpartition_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let xs = (xs + shortcut)?;
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.residual_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let out = if let Some(t) = timing.as_deref_mut() {
            let norm2_start = Instant::now();
            let mlp = xs.apply(&self.norm2)?;
            sync_for_timing(&mlp)?;
            t.norm2_s += norm2_start.elapsed().as_secs_f64();

            let mlp = timed_linear(
                &self.mlp.lin1,
                &mlp,
                &mut t.mlp_lin1_s,
                &mut t.mlp_lin1_matmul_s,
                &mut t.mlp_lin1_bias_s,
            )?;

            let activation_start = Instant::now();
            let mlp = mlp.apply(&self.mlp.activation)?;
            sync_for_timing(&mlp)?;
            t.mlp_activation_s += activation_start.elapsed().as_secs_f64();

            let mlp = timed_linear(
                &self.mlp.lin2,
                &mlp,
                &mut t.mlp_lin2_s,
                &mut t.mlp_lin2_matmul_s,
                &mut t.mlp_lin2_bias_s,
            )?;

            let residual_start = Instant::now();
            let out = (&xs + mlp)?;
            sync_for_timing(&out)?;
            t.mlp_residual_s += residual_start.elapsed().as_secs_f64();
            out
        } else {
            (&xs + xs.apply(&self.norm2)?.apply(&self.mlp)?)?
        };
        if timing.is_some() {
            sync_for_timing(&out)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.norm2_mlp_s += t0.elapsed().as_secs_f64();
            t.total_s += total_start.elapsed().as_secs_f64();
        }
        Ok(out)
    }
}

#[derive(Debug)]
pub struct ImageEncoderViT {
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    neck_conv1: candle_nn::Conv2d,
    neck_ln1: super::LayerNorm2d,
    neck_conv2: candle_nn::Conv2d,
    neck_ln2: super::LayerNorm2d,
    pos_embed: Option<Tensor>,
    span: tracing::Span,
}

impl ImageEncoderViT {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        out_chans: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        use_abs_pos: bool,
        window_size: usize,
        global_attn_indexes: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            in_chans,
            embed_dim,
            patch_size,
            patch_size,
            0,
            vb.pp("patch_embed"),
        )?;
        let mut blocks = Vec::with_capacity(depth);
        let vb_b = vb.pp("blocks");
        for i in 0..depth {
            let window_size = if global_attn_indexes.contains(&i) {
                0
            } else {
                window_size
            };
            let block = Block::new(
                embed_dim,
                num_heads,
                qkv_bias,
                use_rel_pos,
                window_size,
                (img_size / patch_size, img_size / patch_size),
                vb_b.pp(i),
            )?;
            blocks.push(block)
        }
        let neck_conv1 = candle_nn::conv2d_no_bias(
            embed_dim,
            out_chans,
            1,
            Default::default(),
            vb.pp("neck.0"),
        )?;
        let neck_ln1 = super::LayerNorm2d::new(out_chans, 1e-6, vb.pp("neck.1"))?;
        let cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let neck_conv2 = candle_nn::conv2d_no_bias(out_chans, out_chans, 3, cfg, vb.pp("neck.2"))?;
        let neck_ln2 = super::LayerNorm2d::new(out_chans, 1e-6, vb.pp("neck.3"))?;
        let pos_embed = if use_abs_pos {
            let p = vb.get(
                (1, img_size / patch_size, img_size / patch_size, embed_dim),
                "pos_embed",
            )?;
            Some(p)
        } else {
            None
        };
        let span = tracing::span!(tracing::Level::TRACE, "image-encoder-vit");
        Ok(Self {
            patch_embed,
            blocks,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
            pos_embed,
            span,
        })
    }
}

impl Module for ImageEncoderViT {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_impl(xs, None)
    }
}

impl ImageEncoderViT {
    pub fn forward_with_timing(
        &self,
        xs: &Tensor,
        timing: &mut ImageEncoderViTTiming,
    ) -> Result<Tensor> {
        self.forward_impl(xs, Some(timing))
    }

    fn forward_impl(
        &self,
        xs: &Tensor,
        mut timing: Option<&mut ImageEncoderViTTiming>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let total_start = Instant::now();
        let t0 = Instant::now();
        let xs = self.patch_embed.forward(xs)?;
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.patch_embed_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let mut xs = match &self.pos_embed {
            Some(pos_embed) => xs.broadcast_add(pos_embed)?,
            None => xs,
        };
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.pos_embed_s += t0.elapsed().as_secs_f64();
            t.blocks.resize_with(self.blocks.len(), Default::default);
        }
        let t0 = Instant::now();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            xs = if let Some(t) = timing.as_deref_mut() {
                block.forward_with_timing(&xs, &mut t.blocks[block_idx])?
            } else {
                block.forward(&xs)?
            }
        }
        if timing.is_some() {
            sync_for_timing(&xs)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.blocks_s += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let out = xs
            .permute((0, 3, 1, 2))?
            .apply(&self.neck_conv1)?
            .apply(&self.neck_ln1)?
            .apply(&self.neck_conv2)?
            .apply(&self.neck_ln2)?;
        if timing.is_some() {
            sync_for_timing(&out)?;
        }
        if let Some(t) = timing.as_deref_mut() {
            t.neck_s += t0.elapsed().as_secs_f64();
            t.total_s += total_start.elapsed().as_secs_f64();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::Add3Inplace;
    use super::ImageEncoderViT;
    use candle::{DType, Device, Result, Tensor};
    use candle_nn::{Module, VarBuilder};

    fn image_encoder_abs_pos_supports_batched_inputs(device: &Device) -> Result<()> {
        let img_size = 8;
        let patch_size = 4;
        let in_chans = 3;
        let embed_dim = 4;
        let out_chans = 2;
        let vb = VarBuilder::zeros(DType::F32, device);
        let encoder = ImageEncoderViT::new(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            0,
            1,
            out_chans,
            true,
            false,
            true,
            0,
            &[],
            vb,
        )?;
        let xs = Tensor::zeros((2, in_chans, img_size, img_size), DType::F32, device)?;
        let ys = encoder.forward(&xs)?;
        assert_eq!(
            ys.dims(),
            &[2, out_chans, img_size / patch_size, img_size / patch_size]
        );
        Ok(())
    }

    #[test]
    fn image_encoder_abs_pos_supports_batched_inputs_cpu() -> Result<()> {
        image_encoder_abs_pos_supports_batched_inputs(&Device::Cpu)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn image_encoder_abs_pos_supports_batched_inputs_cuda() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping CUDA test: {err}");
                return Ok(());
            }
        };
        image_encoder_abs_pos_supports_batched_inputs(&device)
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn add3_inplace_cuda_matches_broadcast_expression() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("skipping CUDA test: {err}");
                return Ok(());
            }
        };
        let (b, q_h, q_w, k_h, k_w) = (2, 3, 2, 3, 2);
        let attn_len = b * q_h * q_w * k_h * k_w;
        let rel_h_len = b * q_h * q_w * k_h;
        let rel_w_len = b * q_h * q_w * k_w;
        let attn_data: Vec<f32> = (0..attn_len).map(|v| v as f32 / 10.0).collect();
        let rel_h_data: Vec<f32> = (0..rel_h_len).map(|v| v as f32 / 100.0).collect();
        let rel_w_data: Vec<f32> = (0..rel_w_len).map(|v| -(v as f32) / 200.0).collect();

        let attn = Tensor::from_vec(attn_data, (b, q_h, q_w, k_h, k_w), &device)?;
        let rel_h = Tensor::from_vec(rel_h_data, (b, q_h, q_w, k_h), &device)?;
        let rel_w = Tensor::from_vec(rel_w_data, (b, q_h, q_w, k_w), &device)?;
        let expected = (&attn + rel_h.unsqueeze(4)?.broadcast_add(&rel_w.unsqueeze(3)?)?)?;

        attn.inplace_op3(&rel_h, &rel_w, &Add3Inplace(b, q_h, q_w, k_h, k_w))?;
        device.synchronize()?;

        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = attn.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(expected.len(), actual.len());
        for (idx, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < 1e-5,
                "idx {idx}: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }
}
