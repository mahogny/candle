//! Convolution Layers.
use crate::BatchNorm;
use candle::{conv::CudnnFwdAlgo, Result, Tensor};

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
    use super::{conv3d_no_bias, Conv3d, Conv3dConfig};
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
}
