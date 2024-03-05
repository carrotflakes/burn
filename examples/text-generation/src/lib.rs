#[macro_use]
extern crate derive_new;

mod data;
mod model;

pub mod inference;
pub mod training;
pub use data::DbPediaDataset;

#[cfg(feature = "bitnet")]
pub mod transformer {
    pub use burn::nn::{
        bit_transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        BitLinear as Linear, BitLinearConfig as LinearConfig,
    };

    pub const LEARNING_RATE: f64 = 4.0;
    pub const WARMUP_STEPS: usize = 100;
}

#[cfg(not(feature = "bitnet"))]
pub mod transformer {
    pub use burn::nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Linear, LinearConfig,
    };

    pub const LEARNING_RATE: f64 = 0.01;
    pub const WARMUP_STEPS: usize = 6000;
}
