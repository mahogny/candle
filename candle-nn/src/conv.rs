//! Convolution Layers.
use crate::BatchNorm;
use candle::{conv::CudnnFwdAlgo, DType, Result, Storage, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &Conv1dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d_with_algo(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
            self.config.cudnn_fwd_algo,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose1dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Default for ConvTranspose1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose1dConfig,
}

impl ConvTranspose1d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose1dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &ConvTranspose1dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv_transpose1d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dConfig {
    pub padding: [usize; 3],
    pub stride: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: [0; 3],
            stride: [1; 3],
            dilation: [1; 3],
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &Conv2dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn absorb_bn(&self, bn: &BatchNorm) -> Result<Self> {
        if let Some((w_bn, b_bn)) = bn.weight_and_bias() {
            let std_ = w_bn.div(&((bn.running_var() + bn.eps())?.sqrt()?))?;
            let weight = self
                .weight()
                .broadcast_mul(&(std_.reshape((self.weight().dims4()?.0, 1, 1, 1))?))?;
            let bias = match &self.bias {
                None => b_bn.sub(&(std_.mul(bn.running_mean())?))?,
                Some(bias) => b_bn.add(&(std_.mul(&bias.sub(bn.running_mean())?)?))?,
            };
            Ok(Self {
                weight,
                bias: Some(bias),
                config: self.config,
            })
        } else {
            candle::bail!("batch norm does not have weight_and_bias")
        }
    }

    /// Fuses a strict contiguous CPU-f32 1x1 convolution's bias into the
    /// convolution destination. Other layouts and configurations use the
    /// normal module forward unchanged.
    pub fn forward_fused_bias_cpu(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var_os("CANDLE_CPU_CONV1X1_FUSED").is_none_or(|value| value != "0") {
            if let Some(bias) = &self.bias {
                if let Some(output) = conv2d_1x1_f32_bias(x, &self.weight, bias, self.config)? {
                    return Ok(output);
                }
            }
        }
        crate::Module::forward(self, x)
    }
}

fn conv2d_1x1_f32_bias(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    config: Conv2dConfig,
) -> Result<Option<Tensor>> {
    use gemm::{gemm, Parallelism};
    if !x.device().is_cpu()
        || x.dtype() != DType::F32
        || weight.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || config.padding != 0
        || config.stride != 1
        || config.dilation != 1
        || config.groups != 1
        || weight.dims4()?.2 != 1
        || weight.dims4()?.3 != 1
    {
        return Ok(None);
    }
    let (batch, channels, height, width) = x.dims4()?;
    let pixels = height * width;
    let (out_channels, weight_channels, _, _) = weight.dims4()?;
    if channels != weight_channels
        || bias.dims1()? != out_channels
        || batch == 0
        || pixels == 0
        || out_channels == 0
    {
        return Ok(None);
    }
    // Hiera hands FPN a logical NCHW view of its channels-last activations.
    // Its spatial pixels are still regularly strided, so GEMM can consume the
    // `(channel, pixel)` matrix directly instead of materializing a 1x1-conv
    // input transpose. Keep the fast path narrow to layouts whose flattened
    // spatial dimension has one constant stride.
    let x_layout = x.layout();
    let x_stride = x_layout.stride();
    if x_stride.len() != 4 || x_stride[2] != width * x_stride[3] {
        return Ok(None);
    }
    let nchw_contiguous = x_stride[3] == 1
        && x_stride[2] == width
        && x_stride[1] == pixels
        && x_stride[0] == channels * pixels;
    if !nchw_contiguous
        && std::env::var_os("CANDLE_CPU_CONV1X1_STRIDED").is_some_and(|value| value == "0")
    {
        return Ok(None);
    }
    let x_start = x_layout.start_offset();
    let Some((w_start, w_end)) = weight.layout().contiguous_offsets() else {
        return Ok(None);
    };
    let Some((b_start, b_end)) = bias.layout().contiguous_offsets() else {
        return Ok(None);
    };
    let (x_storage, _) = x.storage_and_layout();
    let (weight_storage, _) = weight.storage_and_layout();
    let (bias_storage, _) = bias.storage_and_layout();
    let (Storage::Cpu(x_storage), Storage::Cpu(weight_storage), Storage::Cpu(bias_storage)) =
        (&*x_storage, &*weight_storage, &*bias_storage)
    else {
        return Ok(None);
    };
    let input = &x_storage.as_slice::<f32>()?[x_start..];
    let weights = &weight_storage.as_slice::<f32>()?[w_start..w_end];
    let biases = &bias_storage.as_slice::<f32>()?[b_start..b_end];
    let input_last = (batch - 1) * x_stride[0]
        + (channels - 1) * x_stride[1]
        + (height - 1) * x_stride[2]
        + (width - 1) * x_stride[3];
    if input_last >= input.len()
        || weights.len() != out_channels * channels
        || biases.len() != out_channels
    {
        return Ok(None);
    }
    let output_len = batch * out_channels * pixels;
    let mut output;
    if std::env::var_os("CANDLE_CPU_CONV1X1_SINGLE_WRITE_SEED").is_none_or(|value| value != "0") {
        output = Vec::with_capacity(output_len);
        output.spare_capacity_mut()[..output_len]
            .par_chunks_mut(pixels)
            .enumerate()
            .for_each(|(plane_index, plane)| {
                let bias = biases[plane_index % out_channels];
                for output in plane {
                    output.write(bias);
                }
            });
        // SAFETY: each disjoint channel plane above writes all its elements
        // before the vector length is increased.
        unsafe { output.set_len(output_len) };
    } else {
        output = vec![0f32; output_len];
        output
            .par_chunks_mut(out_channels * pixels)
            .for_each(|batch_output| {
                for (channel, bias) in biases.iter().enumerate() {
                    batch_output[channel * pixels..(channel + 1) * pixels].fill(*bias);
                }
            });
    }
    let parallelism = if rayon::current_num_threads() > 1 {
        Parallelism::Rayon(rayon::current_num_threads())
    } else {
        Parallelism::None
    };
    for batch_index in 0..batch {
        let output = &mut output
            [batch_index * out_channels * pixels..(batch_index + 1) * out_channels * pixels];
        unsafe {
            gemm(
                out_channels,
                pixels,
                channels,
                output.as_mut_ptr(),
                1,
                pixels as isize,
                true,
                weights.as_ptr(),
                1,
                channels as isize,
                input.as_ptr().add(batch_index * x_stride[0]),
                x_stride[3] as isize,
                x_stride[1] as isize,
                1f32,
                1f32,
                false,
                false,
                false,
                parallelism,
            );
        }
    }
    Ok(Some(Tensor::from_vec(
        output,
        (batch, out_channels, height, width),
        x.device(),
    )?))
}

impl crate::Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv2d_with_algo(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
            self.config.cudnn_fwd_algo,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv3dConfig,
}

impl Conv3d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: Conv3dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &Conv3dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl crate::Module for Conv3d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv3d_with_algo(
            &self.weight,
            self.config.padding,
            self.config.stride,
            self.config.dilation,
            self.config.groups,
            self.config.cudnn_fwd_algo,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvTranspose2dConfig {
    pub padding: usize,
    pub output_padding: usize,
    pub stride: usize,
    pub dilation: usize,
    // TODO: support groups.
}

impl Default for ConvTranspose2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            output_padding: 0,
            stride: 1,
            dilation: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConvTranspose2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvTranspose2dConfig,
}

impl ConvTranspose2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: ConvTranspose2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &ConvTranspose2dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Inference-only CPU f32 bias fusion for non-overlapping 2x2/stride-2
    /// transposed convolutions. Other layouts and devices use `forward`.
    pub fn forward_fused_bias_cpu(&self, x: &Tensor) -> Result<Tensor> {
        if std::env::var_os("CANDLE_CONVTRANSPOSE_FUSED_BIAS").is_none_or(|value| value != "0") {
            if let Some(bias) = &self.bias {
                if let Some(output) = x.conv_transpose2d_bias_cpu(
                    &self.weight,
                    bias,
                    self.config.padding,
                    self.config.output_padding,
                    self.config.stride,
                    self.config.dilation,
                )? {
                    return Ok(output);
                }
            }
        }
        crate::Module::forward(self, x)
    }

    /// Inference-only CPU f32 fusion of transposed convolution, bias and an
    /// equal-shaped residual. Falls back to the composed operations outside
    /// the non-overlapping 2x2/stride-2 specialization.
    pub fn forward_add_residual_cpu(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        if std::env::var_os("CANDLE_CONVTRANSPOSE_FUSED_RESIDUAL").is_none_or(|value| value != "0")
        {
            if let Some(bias) = &self.bias {
                if let Some(output) = x.conv_transpose2d_bias_residual_cpu(
                    &self.weight,
                    bias,
                    residual,
                    self.config.padding,
                    self.config.output_padding,
                    self.config.stride,
                    self.config.dilation,
                )? {
                    return Ok(output);
                }
            }
        }
        self.forward_fused_bias_cpu(x)?.broadcast_add(residual)
    }
}

impl crate::Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv_transpose2d(
            &self.weight,
            self.config.padding,
            self.config.output_padding,
            self.config.stride,
            self.config.dilation,
        )?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

pub fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv1d::new(ws, Some(bs), cfg))
}

pub fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv1d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight",
        init_ws,
    )?;
    Ok(Conv1d::new(ws, None, cfg))
}

pub fn conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1. / (out_channels as f64 * kernel_size as f64).sqrt();
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels / cfg.groups, kernel_size),
        "weight",
        init,
    )?;
    let bs = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(ConvTranspose1d::new(ws, Some(bs), cfg))
}

pub fn conv_transpose1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose1d> {
    let bound = 1. / (out_channels as f64 * kernel_size as f64).sqrt();
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels / cfg.groups, kernel_size),
        "weight",
        init,
    )?;
    Ok(ConvTranspose1d::new(ws, None, cfg))
}

pub fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub fn conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    Ok(Conv2d::new(ws, None, cfg))
}

pub fn conv3d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv3dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv3d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv3d::new(ws, Some(bs), cfg))
}

pub fn conv3d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv3dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv3d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
            kernel_size,
        ),
        "weight",
        init_ws,
    )?;
    Ok(Conv3d::new(ws, None, cfg))
}

pub fn conv_transpose2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose2dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose2d> {
    let bound = 1. / (out_channels as f64).sqrt() / kernel_size as f64;
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels, kernel_size, kernel_size),
        "weight",
        init,
    )?;
    let bs = vb.get_with_hints(out_channels, "bias", init)?;
    Ok(ConvTranspose2d::new(ws, Some(bs), cfg))
}

pub fn conv_transpose2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose2dConfig,
    vb: crate::VarBuilder,
) -> Result<ConvTranspose2d> {
    let bound = 1. / (out_channels as f64).sqrt() / kernel_size as f64;
    let init = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let ws = vb.get_with_hints(
        (in_channels, out_channels, kernel_size, kernel_size),
        "weight",
        init,
    )?;
    Ok(ConvTranspose2d::new(ws, None, cfg))
}

#[cfg(test)]
mod tests {
    use super::{
        conv3d_no_bias, Conv2d, Conv2dConfig, Conv3d, Conv3dConfig, ConvTranspose2d,
        ConvTranspose2dConfig,
    };
    use candle::{DType, Device, Module, Result, Tensor};

    #[test]
    fn conv3d_forward_adds_bias() -> Result<()> {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((1, 1, 1, 1, 1), DType::F32, dev)?;
        let bias = Tensor::new(&[2f32], dev)?;
        let layer = Conv3d::new(weight, Some(bias), Conv3dConfig::default());
        let input = Tensor::new((0..8).map(|v| v as f32).collect::<Vec<_>>(), dev)?
            .reshape((1, 1, 2, 2, 2))?;

        let output = layer.forward(&input)?;
        assert_eq!(output.dims(), &[1, 1, 2, 2, 2]);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            &[2., 3., 4., 5., 6., 7., 8., 9.]
        );

        Ok(())
    }

    #[test]
    fn conv3d_no_bias_uses_config_and_weight_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let cfg = Conv3dConfig {
            padding: [1, 0, 2],
            stride: [1, 2, 2],
            dilation: [1, 1, 1],
            groups: 2,
            cudnn_fwd_algo: None,
        };
        let vb = crate::VarBuilder::zeros(DType::F32, dev);
        let layer = conv3d_no_bias(4, 6, 3, cfg, vb)?;

        assert_eq!(layer.weight().dims(), &[6, 2, 3, 3, 3]);
        assert!(layer.bias().is_none());
        assert_eq!(layer.config(), &cfg);

        Ok(())
    }

    #[test]
    fn fused_cpu_1x1_bias_matches_composed_conv() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::new((0..24).map(|v| v as f32 * 0.1).collect::<Vec<_>>(), dev)?
            .reshape((1, 3, 2, 4))?;
        let weight =
            Tensor::new(&[0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6], dev)?.reshape((2, 3, 1, 1))?;
        let bias = Tensor::new(&[0.7f32, -0.8], dev)?;
        let layer = Conv2d::new(weight, Some(bias), Conv2dConfig::default());
        let expected = layer.forward(&input)?;
        let actual = layer.forward_fused_bias_cpu(&input)?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        for (expected, actual) in expected.iter().zip(actual) {
            assert!((expected - actual).abs() <= 1e-6);
        }
        Ok(())
    }

    #[test]
    fn fused_cpu_1x1_bias_matches_composed_conv_for_strided_nchw() -> Result<()> {
        let dev = &Device::Cpu;
        // Hiera/FPN exposes this logical NCHW shape from channels-last
        // storage. The test keeps the physical NHWC layout and permutes it
        // into the view consumed by the strided GEMM fast path.
        let input = Tensor::new((0..24).map(|v| v as f32 * 0.1).collect::<Vec<_>>(), dev)?
            .reshape((1, 2, 4, 3))?
            .permute((0, 3, 1, 2))?;
        assert!(!input.is_contiguous());
        let weight =
            Tensor::new(&[0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6], dev)?.reshape((2, 3, 1, 1))?;
        let bias = Tensor::new(&[0.7f32, -0.8], dev)?;
        let layer = Conv2d::new(weight, Some(bias), Conv2dConfig::default());
        let expected = layer.forward(&input)?;
        let actual = layer.forward_fused_bias_cpu(&input)?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        for (expected, actual) in expected.iter().zip(actual) {
            assert!((expected - actual).abs() <= 1e-6);
        }
        Ok(())
    }

    #[test]
    fn fused_cpu_transpose2d_bias_matches_composed_conv_for_strided_nchw() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::from_vec(
            (0..36).map(|index| index as f32 * 0.07 - 0.9).collect(),
            (2, 2, 3, 3),
            dev,
        )?
        .permute((0, 3, 1, 2))?;
        assert!(!input.is_contiguous());
        let weight = Tensor::from_vec(
            (0..48)
                .map(|index| (index % 11) as f32 * 0.09 - 0.4)
                .collect(),
            (3, 4, 2, 2),
            dev,
        )?;
        let bias = Tensor::new(&[0.7f32, -0.8, 0.25, -0.125], dev)?;
        let layer = ConvTranspose2d::new(
            weight.clone(),
            Some(bias.clone()),
            ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        let expected = layer.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = input
            .conv_transpose2d_bias_cpu(&weight, &bias, 0, 0, 2, 1)?
            .expect("eligible CPU f32 transposed convolution")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert!((expected - actual).abs() <= 1e-6);
        }
        Ok(())
    }

    #[test]
    fn fused_cpu_transpose2d_residual_matches_composed_conv() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::from_vec(
            (0..36).map(|index| index as f32 * 0.07 - 0.9).collect(),
            (2, 2, 3, 3),
            dev,
        )?
        .permute((0, 3, 1, 2))?;
        let weight = Tensor::from_vec(
            (0..48)
                .map(|index| (index % 11) as f32 * 0.09 - 0.4)
                .collect(),
            (3, 4, 2, 2),
            dev,
        )?;
        let bias = Tensor::new(&[0.7f32, -0.8, 0.25, -0.125], dev)?;
        let residual = Tensor::from_vec(
            (0..96).map(|index| index as f32 * 0.003 - 0.2).collect(),
            (1, 4, 4, 6),
            dev,
        )?;
        let layer = ConvTranspose2d::new(
            weight.clone(),
            Some(bias.clone()),
            ConvTranspose2dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        let expected = layer
            .forward(&input)?
            .broadcast_add(&residual)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let actual = input
            .conv_transpose2d_bias_residual_cpu(&weight, &bias, &residual, 0, 0, 2, 1)?
            .expect("eligible CPU f32 transposed convolution")
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert!((expected - actual).abs() <= 1e-6);
        }
        Ok(())
    }
}
