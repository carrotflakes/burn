use alloc::vec::Vec;

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_core as burn;
use burn_tensor::Data;

/// Configuration to create an [GrayEncoding](GrayEncoding) layer.
#[derive(Config)]
pub struct GrayEncodingConfig {
    /// Maximum sequence size to use.
    #[config(default = "5_000")]
    max_sequence_size: usize,

    /// The size of each vector.
    d_model: usize,
}

/// Gray encoding layer for transformer models.
#[derive(Module, Debug)]
pub struct GrayEncoding<B: Backend> {
    encoding: Tensor<B, 3>,
}

impl GrayEncodingConfig {
    /// Initialize a new [GrayEncoding](GrayEncoding) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GrayEncoding<B> {
        let encoding =
            generate::<B>(self.max_sequence_size, self.d_model, &device).unsqueeze::<3>();

        GrayEncoding { encoding }
    }
}

impl<B: Backend> GrayEncoding<B> {
    /// Applies the forward pass on the input tensor by adding the gray code to the input.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq_length, d_model_input] = input.dims();

        let [batch_size, max_sequence_size, d_model] = self.encoding.dims();

        assert!(
            max_sequence_size >= seq_length,
            "max_sequence_size({}) must be greater or equal than length({seq_length})",
            max_sequence_size,
        );

        assert!(
            d_model_input == d_model,
            "d_model({}) of the input must be equal to d_model of encoding({})",
            d_model_input,
            d_model,
        );

        let slices = [0..batch_size, 0..seq_length, 0..d_model];

        input.add(self.encoding.clone().slice(slices))
    }
}

/// Returns gray code for positional embedding introduced in
pub fn generate<B: Backend>(length: usize, d_model: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = Vec::with_capacity(length);

    for i in 0..length {
        let mut row = Vec::with_capacity(d_model);

        for k in 0..d_model {
            row.push(if (i >> k & 1) ^ (i >> (k + 1) & 1) == 1 {
                1.0
            } else {
                -1.0
            });
        }

        data.push(row);
    }

    let data = Data::new(
        data.into_iter().flatten().collect(),
        [length, d_model].into(),
    );

    Tensor::<B, 2>::from_data(data.convert(), device)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_module() {
        let d_model = 3;
        let length = 6;

        let batch_size = 2;

        let device = Default::default();
        let pe = GrayEncodingConfig::new(d_model).init::<TestBackend>(&device);

        let tensor = Tensor::zeros([batch_size, length, d_model], &device);

        let output = pe.forward(tensor);

        assert_eq!(output.shape().dims, [batch_size, length, d_model]);

        let p = 1.0;
        let n = -1.0;
        let expected = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [n, n, n],
                    [p, n, n],
                    [p, p, n],
                    [n, p, n],
                    [n, p, p],
                    [p, p, p],
                ],
                [
                    [n, n, n],
                    [p, n, n],
                    [p, p, n],
                    [n, p, n],
                    [n, p, p],
                    [p, p, p],
                ],
            ],
            &device,
        );

        output.to_data().assert_approx_eq(&expected.to_data(), 5);
    }
}
