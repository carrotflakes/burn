use crate::{
    data::{Gpt2Tokenizer, TextGenerationBatch, TextGenerationBatcher, Tokenizer},
    model::TextGenerationModelConfig,
    training::ExperimentConfig,
};
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use std::sync::Arc;

// Define inference function
pub fn infer<B: Backend>(
    //, D: TextClassificationDataset + 'static
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    text: String,      // Text for inference
    len: usize,        // Length of output text
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(Gpt2Tokenizer::default());

    // Initialize batcher for batching samples
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = TextGenerationModelConfig::new(
        config.transformer,
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        config.max_seq_length,
    )
    .init_with::<B>(record);

    // Run inference on the given text samples
    println!("Running inference ...");
    let mut input_tokens = tokenizer.encode(&text, true);
    let mut output_tokens = Vec::new();
    for _ in 0..len {
        let item: TextGenerationBatch<B> = batcher.batch_from_tokens(&input_tokens);
        let token_len = item.tokens.dims()[1];
        let predictions = model.infer(item); // Get model predictions
        let output =
            predictions.argmax(2).into_data().convert::<i32>().value[token_len - 1] as usize;
        input_tokens.push(output);
        output_tokens.push(output);
    }

    // Print the output text
    let output_text = tokenizer.decode(&output_tokens);
    println!("Output: {}", output_text);
}
