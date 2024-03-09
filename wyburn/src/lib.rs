extern crate alloc;

pub mod nn;

#[cfg(test)]
pub type TestBackend = burn_ndarray::NdArray<f32>;
