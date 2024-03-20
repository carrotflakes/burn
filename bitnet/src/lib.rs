#[macro_use]
extern crate derive_new;

pub mod burn_model;
pub mod data;
pub mod model;
pub mod model_float;

use crate::data::{Gpt2Tokenizer, Tokenizer};
use burn::{
    backend::NdArray,
    nn::{bit_transformer::TransformerEncoderConfig, LayerNormRecord},
    optim::BitAdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use std::sync::Arc;

type B = NdArray;

pub fn run(
    //, D: TextClassificationDataset + 'static
    // device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    text: String,       // Text for inference
    len: usize,         // Length of output text
) {
    let device = burn::tensor::Device::<B>::Cpu;

    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(Gpt2Tokenizer::default());

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record: burn_model::TextGenerationModelRecord<NdArray> = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    // let model = TextGenerationModelConfig::new(
    //     config.transformer,
    //     tokenizer.vocab_size(),
    //     tokenizer.pad_token(),
    //     config.max_seq_length,
    // )
    // .init_with::<B>(record);

    fn lin(
        d_in: usize,
        d_out: usize,
        record: &Vec<f32>,
        ln: &Option<LayerNormRecord<B>>,
    ) -> model::Linear {
        let gamma = record.iter().map(|x| x.abs()).sum::<f32>() / record.len() as f32;
        let w: Vec<_> = record.iter().map(|x| (x / gamma).round()).collect();
        model::Linear::from_weights(
            d_in,
            d_out,
            &w,
            gamma,
            ln.as_ref().map(|ln| {
                model::LayerNorm::new(
                    ln.gamma.val().into_data().value,
                    ln.beta.val().into_data().value,
                )
            }),
        )
    }

    let model = model::BitNet::new(
        record
            .embedding_token
            .weight
            .val()
            .into_data()
            .value
            .chunks(config.transformer.d_model)
            // .map(|x| x.iter().map(|x| (x.clamp(-1.0, 1.0) * 127.0).round() as i8).collect())
            .map(|x| x.to_vec())
            .collect(),
        record.embedding_pos.weight.val().into_data().value,
        record
            .transformer
            .layers
            .iter()
            .map(|layer| {
                model::Layer::new(
                    model::MultiHeadAttention::new(
                        config.transformer.n_heads,
                        lin(
                            config.transformer.d_model,
                            config.transformer.d_model,
                            &layer.mha.query.weight.val().into_data().value,
                            &layer.mha.query.norm,
                        ),
                        lin(
                            config.transformer.d_model,
                            config.transformer.d_model,
                            &layer.mha.key.weight.val().into_data().value,
                            &layer.mha.key.norm,
                        ),
                        lin(
                            config.transformer.d_model,
                            config.transformer.d_model,
                            &layer.mha.value.weight.val().into_data().value,
                            &layer.mha.value.norm,
                        ),
                        lin(
                            config.transformer.d_model,
                            config.transformer.d_model,
                            &layer.mha.output.weight.val().into_data().value,
                            &layer.mha.output.norm,
                        ),
                    ),
                    model::PositionWiseFeedForward::new(
                        lin(
                            config.transformer.d_model,
                            config.transformer.d_ff,
                            &layer.pwff.linear_inner.weight.val().into_data().value,
                            &layer.pwff.linear_inner.norm,
                        ),
                        lin(
                            config.transformer.d_ff,
                            config.transformer.d_model,
                            &layer.pwff.linear_outer.weight.val().into_data().value,
                            &layer.pwff.linear_outer.norm,
                        ),
                    ),
                    model::LayerNorm::new(
                        layer.norm_2.gamma.val().into_data().value,
                        layer.norm_2.beta.val().into_data().value,
                    ),
                    model::LayerNorm::new(
                        layer.norm_1.gamma.val().into_data().value,
                        layer.norm_1.beta.val().into_data().value,
                    ),
                )
            })
            .collect(),
        lin(
            config.transformer.d_model,
            tokenizer.vocab_size(),
            &record.output.weight.val().into_data().value,
            &record.output.norm,
        ),
    );

    // Run inference on the given text samples
    println!("Running inference ...");
    let t = std::time::Instant::now();
    let mut input_tokens = tokenizer.encode(&text, false);
    let mut output_tokens = Vec::new();
    let mut states = model::BitNetState::vec_from_tokens(&input_tokens);
    for _ in 0..len {
        model::BitNetState::vec_add(&mut states, &input_tokens);
        let logits = model::BitNetState::vec_compute(&mut states, &model, &input_tokens);
        let max = logits
            .iter()
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let output = logits.iter().position(|x| x == max).unwrap();
        input_tokens.push(output);
        output_tokens.push(output);
    }
    println!("Elapsed: {:?}", t.elapsed());

    // Print the output text
    let output_text = tokenizer.decode(&output_tokens);
    println!("Output: {}", output_text);
}

#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    optimizer: BitAdamConfig,
    #[config(default = 512)]
    pub max_seq_length: usize,
    #[config(default = 6)]
    batch_size: usize,
    #[config(default = 50)]
    num_epochs: usize,
}
