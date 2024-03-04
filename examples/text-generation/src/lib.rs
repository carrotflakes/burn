#[macro_use]
extern crate derive_new;

mod data;
mod model;

pub mod training;
pub use data::DbPediaDataset;

#[cfg(feature = "bitnet")]
pub mod transformer {
    pub use burn::nn::{
        bit_transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        BitLinear as Linear, BitLinearConfig as LinearConfig,
    };

    pub const LEARNING_RATE: f64 = 0.1;
}

#[cfg(not(feature = "bitnet"))]
pub mod transformer {
    pub use burn::nn::{
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Linear, LinearConfig,
    };

    pub const LEARNING_RATE: f64 = 0.01;
}
