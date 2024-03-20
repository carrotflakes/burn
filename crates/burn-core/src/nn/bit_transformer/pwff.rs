use crate as burn;

use crate::nn::{BitLinear, BitLinearConfig, Initializer};
use crate::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Gelu},
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [position-wise feed-forward](PositionWiseFeedForward) layer.
#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    /// The size of the input and output features.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub d_ff: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies the position-wise feed-forward network to the input tensor.
///
/// # Params
///
/// - linear inner: Linear layer with `d_model` input features and `d_ff` output features.
/// - linear outer: Linear layer with `d_ff` input features and `d_model` output features.
#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: BitLinear<B>,
    linear_outer: BitLinear<B>,
    dropout: Dropout,
    gelu: Gelu,
}

impl PositionWiseFeedForwardConfig {
    /// Initialize a new [position-wise feed-forward](PositionWiseFeedForward) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: BitLinearConfig::new(self.d_model, self.d_ff)
                .with_initializer(self.initializer.clone())
                .with_input_norm(false)
                .init(device),
            linear_outer: BitLinearConfig::new(self.d_ff, self.d_model)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            gelu: Gelu::new(),
        }
    }
    /// Initialize a new [position-wise feed-forward](PositionWiseFeedForward) module with a
    /// [record](PositionWiseFeedForwardRecord).
    pub fn init_with<B: Backend>(
        &self,
        record: PositionWiseFeedForwardRecord<B>,
    ) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: BitLinearConfig::new(self.d_model, self.d_ff)
                .with_input_norm(false)
                .init_with(record.linear_inner),
            linear_outer: BitLinearConfig::new(self.d_ff, self.d_model)
                .init_with(record.linear_outer),
            dropout: DropoutConfig::new(self.dropout).init(),
            gelu: Gelu::new(),
        }
    }
}

impl<B: Backend> PositionWiseFeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - tensor: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
