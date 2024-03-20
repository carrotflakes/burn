#[cfg(feature = "f16")]
type Elem = burn::tensor::f16;
#[cfg(not(feature = "f16"))]
type Elem = f32;

type Backend = burn::backend::LibTorch<Elem>;

fn main() {
    if cfg!(feature = "bitnet") {
        log::info!("Using BitNet");
    } else {
        log::info!("Using regular Transformer");
    }

    text_generation::inference::infer::<Backend>(
        if cfg!(target_os = "macos") {
            burn::tensor::Device::<Backend>::Mps
        } else {
            burn::tensor::Device::<Backend>::Cuda(0)
        },
        "/tmp/text-generation",
        "[START]how".to_owned(),
        64,
    );
}
