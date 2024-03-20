use burn::{
    nn::{Embedding, EmbeddingConfig},
    prelude::*,
};

pub use burn::nn::{
    bit_transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    BitLinear as Linear, BitLinearConfig as LinearConfig,
};

#[derive(Config)]
pub struct TextGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    pub transformer: TransformerEncoder<B>,
    pub embedding_token: Embedding<B>,
    pub embedding_pos: Embedding<B>,
    pub output: Linear<B>,
    pub vocab_size: usize,
    pub pad_token: usize,
    pub max_seq_length: usize,
}

impl TextGenerationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextGenerationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

        TextGenerationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_length: self.max_seq_length,
        }
    }

    pub fn init_with<B: Backend>(
        &self,
        record: TextGenerationModelRecord<B>,
    ) -> TextGenerationModel<B> {
        let output =
            LinearConfig::new(self.transformer.d_model, self.vocab_size).init_with(record.output);
        let transformer = self.transformer.init_with(record.transformer);
        let embedding_token = EmbeddingConfig::new(self.vocab_size, self.transformer.d_model)
            .init_with(record.embedding_token);
        let embedding_pos = EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model)
            .init_with(record.embedding_pos);

        TextGenerationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_seq_length: self.max_seq_length,
        }
    }
}
