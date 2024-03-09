use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_core as burn;
use burn_tensor::{Data, Shape};

#[derive(Config)]
pub struct OrderGateConfig {
    size: usize,
    #[config(default = "1.0")]
    hardness: f32,
    #[config(default = "0.5")]
    pivot: f32,
}

#[derive(Module, Debug)]
pub struct OrderGate<B: Backend> {
    indexes: Tensor<B, 1>,
    pivot: Param<Tensor<B, 0>>,
    hardness: f32,
}

impl OrderGateConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> OrderGate<B> {
        let indexes = Tensor::arange(0..self.size as i64, device).float();
        let pivot = Tensor::<B, 0>::from_data(
            Data::new(vec![self.pivot], Shape::new([])).convert(),
            &device,
        );

        OrderGate {
            indexes,
            pivot: Param::from(pivot),
            hardness: self.hardness,
        }
    }
}

impl<B: Backend> OrderGate<B> {
    pub fn pivot(&self) -> &Param<Tensor<B, 0>> {
        &self.pivot
    }

    pub fn forward(&self) -> Tensor<B, 1> {
        let pivot = self.pivot.val().mul_scalar(self.indexes.dims()[0] as f32);
        burn_tensor::activation::sigmoid(
            (self.indexes.clone() - pivot.unsqueeze()).mul_scalar(self.hardness),
        )
    }
}

pub fn lerp<const N: usize, B: Backend>(
    a: Tensor<B, N>,
    b: Tensor<B, N>,
    t: Tensor<B, 1>,
) -> Tensor<B, N> {
    a.mul(t.clone().neg().add_scalar(1.0).unsqueeze()) + b.mul(t.unsqueeze())
}
