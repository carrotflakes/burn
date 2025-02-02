use burn::optim::decay::WeightDecayConfig;
use text_generation::{training::ExperimentConfig, DbPediaDataset};

#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::backend::Autodiff<burn::backend::LibTorch<Elem>>;

fn main() {
    if cfg!(feature = "bitnet") {
        log::info!("Using BitNet");
    } else {
        log::info!("Using regular Transformer");
    }

    let config = ExperimentConfig::new(
        text_generation::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(if cfg!(feature = "bitnet") {
                true
            } else {
                true
            }),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    )
    .with_num_epochs(4);

    text_generation::training::train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            burn::tensor::Device::<Backend>::Mps
        } else {
            burn::tensor::Device::<Backend>::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}
