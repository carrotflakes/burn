use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::{backend::Backend, Tensor};
use libm::sqrt;

use super::Initializer;
use super::LayerNorm;
use super::LayerNormConfig;

/// Configuration to create a [BitLinear](BitLinear) layer.
#[derive(Config, Debug)]
pub struct BitLinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[config(default = false)]
    pub bias: bool,
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::KaimingUniform{gain:1.0/sqrt(3.0), fan_out_only:false}")]
    pub initializer: Initializer,
    /// If the input tensor should be normalized.
    #[config(default = true)]
    pub input_norm: bool,
}

/// Applies a linear transformation to the input tensor:
///
/// `O = IW + b`
#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    /// Matrix of shape `[d_input, d_output]` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub weight: Param<Tensor<B, 2>>,
    /// Vector of size `d_output` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub bias: Option<Param<Tensor<B, 1>>>,
    /// Layer normalization to apply to the input tensor.
    pub norm: Option<LayerNorm<B>>,
}

impl BitLinearConfig {
    /// Initialize a new [Bitlinear](BitLinear) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BitLinear<B> {
        let shape = [self.d_input, self.d_output];
        let weight =
            self.initializer
                .init_with(shape, Some(self.d_input), Some(self.d_output), device);
        let bias = if self.bias {
            Some(self.initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };
        let norm = if self.input_norm {
            Some(LayerNormConfig::new(self.d_input).init(device))
        } else {
            None
        };

        BitLinear {
            weight: Param::from(weight),
            bias: bias.map(Param::from),
            norm,
        }
    }

    /// Initialize a new [linear](BitLinear) module with a [record](BitLinearRecord).
    pub fn init_with<B: Backend>(&self, record: BitLinearRecord<B>) -> BitLinear<B> {
        BitLinear {
            weight: record.weight,
            // binarized_weight: todo!(),
            bias: record.bias,
            norm: record
                .norm
                .map(|n| LayerNormConfig::new(self.d_input).init_with(n)),
        }
    }
}

impl<B: Backend> BitLinear<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_input]`
    /// - output: `[..., any, d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let bit = 8;
        let qb = 2.0f32.powi(bit - 1);
        let eps = 1e-5;

        let w = self.weight.val();
        let w_gamma = w.clone().abs().mean().reshape([0usize; 0]); // L1 norm
        let w_ = w.clone().detach() / (w_gamma.clone() + eps).unsqueeze();
        let w_ = w_.round().clamp(-1.0, 1.0);
        let w_ = (w_ - w.clone()).detach() + w.clone(); // STE

        let input = if let Some(norm) = &self.norm {
            norm.forward(input)
        } else {
            input
        };

        return input.clone().matmul(w_.unsqueeze()).mul_scalar(w_gamma.into_scalar());

        let input_gamma = input.clone().abs().max().reshape([0usize; 0]); // L-infinity norm
        let input_ = (input.clone().detach()
            * ((input_gamma.clone() + eps).recip() * qb).unsqueeze())
        .clamp(-qb + eps, qb - eps);
        let input_ = (input_ - input.clone()).detach() + input.clone(); // STE

        let y = input_.clone().matmul(w_.unsqueeze());

        let y = match &self.bias {
            Some(bias) => y + bias.val().unsqueeze(),
            None => y,
        };

        // Dequantize
        let beta = w_gamma;
        let gamma = input_gamma;
        let y = y * (beta * gamma / qb).unsqueeze();
        // let y = y * (gamma / qb).unsqueeze();

        y
    }
}
