pub struct BitNet {
    embedding: Vec<Vec<f32>>,
    pos: Vec<f32>,
    layers: Vec<Layer>,
    output: Linear,
}

impl BitNet {
    pub fn new(
        embedding: Vec<Vec<f32>>,
        pos: Vec<f32>,
        layers: Vec<Layer>,
        output: Linear,
    ) -> Self {
        Self {
            embedding,
            pos,
            layers,
            output,
        }
    }

    pub fn compute(
        &self,
        layer_states_before: &[&[LayerState]],
        token: usize,
        pos: usize,
    ) -> (Vec<LayerState>, Vec<i32>) {
        let mut x: Vec<_> = self.embedding[token].clone();

        // Add positional encoding
        let d = x.len();
        x.iter_mut()
            .zip(self.pos[pos * d..].iter())
            .for_each(|(x, p)| *x = (*x + p) / 2.0);

        let s = 127.0;
        let mut x = x.iter().map(|x| (x * s).round() as i32).collect::<Vec<_>>();

        let mut layer_states = vec![];
        for (j, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            layer.norm_1.forward_int(&mut x);

            let queries = layer.mha.query.forward_int(&x);
            let keys = layer.mha.key.forward_int(&x);
            let values = layer.mha.value.forward_int(&x);

            layer_states.push(LayerState { keys, values });

            let mut values = vec![0; layer.mha.value.d_output];
            let heads = layer.mha.heads;
            for i in 0..heads {
                let range = i * queries.len() / heads..(i + 1) * queries.len() / heads;
                let query = &queries[range.clone()];
                let attn: Vec<i32> = layer_states_before
                    .iter()
                    .map(|layer_states| {
                        layer_states[j].keys[range.clone()]
                            .iter()
                            .zip(query.iter())
                            .map(|(k, q)| *k * *q)
                            .sum::<i32>()
                    })
                    .collect();
                let attn_last = layer_states.last().unwrap().keys[range.clone()]
                    .iter()
                    .zip(query.iter())
                    .map(|(k, q)| *k * *q)
                    .sum::<i32>();
                let scale = 1.0 / ((queries.len() / heads) as f32).sqrt() / 127.0 / 127.0;
                let mut attn: Vec<f32> = attn.iter().map(|x| *x as f32 * scale).collect();
                attn.push(attn_last as f32 * scale);
                softmax_(&mut attn);

                let value = &mut values[range.clone()];

                for (layer_states, attn) in layer_states_before.iter().zip(attn.iter()) {
                    value
                        .iter_mut()
                        .zip(layer_states[j].values[range.clone()].iter())
                        .for_each(|(v, x)| *v += x * (attn * 127.0) as i32);
                }
                value
                    .iter_mut()
                    .zip(layer_states.last().unwrap().values[range.clone()].iter())
                    .for_each(|(v, x)| *v += x * (attn.last().unwrap() * 127.0) as i32);
            }
            x = layer.mha.out.forward_int(&values);

            x.iter_mut().zip(residual.iter()).for_each(|(x, r)| *x += r);
            let residual = x.clone();

            layer.norm_2.forward_int(&mut x);
            x = layer.pwff.forward_int(&x);

            x.iter_mut().zip(residual.iter()).for_each(|(x, r)| *x += r);
        }

        (layer_states, x)
    }
}

pub struct BitNetState {
    token: usize,
    #[allow(dead_code)]
    prob: f32,
    layers: Vec<LayerState>,
    nexts: Vec<BitNetState>,
}

impl BitNetState {
    pub fn vec_from_tokens(inputs: &[usize]) -> Vec<Self> {
        if let Some((&token, inputs)) = inputs.split_first() {
            vec![BitNetState {
                token,
                prob: 1.0,
                layers: Vec::new(),
                nexts: Self::vec_from_tokens(inputs),
            }]
        } else {
            vec![]
        }
    }

    pub fn vec_add(vec: &mut Vec<Self>, inputs: &[usize]) {
        if let Some((&token, inputs)) = inputs.split_first() {
            if let Some(i) = vec.iter().position(|x| x.token == token) {
                Self::vec_add(&mut vec[i].nexts, inputs);
            } else {
                vec.push(BitNetState {
                    token,
                    prob: 1.0,
                    layers: Vec::new(),
                    nexts: Self::vec_from_tokens(inputs),
                });
            }
        }
    }

    pub fn vec_compute(vec: &mut Vec<Self>, bitnet: &BitNet, inputs: &[usize]) -> Vec<i32> {
        let mut layer_states = vec![];
        let x = Self::vec_compute_(vec, bitnet, &mut layer_states, inputs, 0);
        bitnet.output.forward_int(&x)
    }

    pub fn vec_compute_<'b, 'a: 'b>(
        vec: &'a mut Vec<Self>,
        bitnet: &'a BitNet,
        layer_states: &mut Vec<&'b [LayerState]>,
        inputs: &'a [usize],
        depth: usize,
    ) -> Vec<i32> {
        let Some((&token, inputs)) = inputs.split_first() else {
            return vec![];
        };
        if let Some(s) = vec.iter_mut().find(|x| x.token == token) {
            let x = if s.layers.is_empty() {
                let (layers, x) = bitnet.compute(&layer_states, token, depth);
                s.layers = layers;
                x
            } else {
                vec![]
            };
            layer_states.push(&s.layers);
            let r = Self::vec_compute_(&mut s.nexts, bitnet, layer_states, inputs, depth + 1);
            layer_states.pop();
            if r.is_empty() {
                x
            } else {
                r
            }
        } else {
            todo!()
        }
    }
}

pub struct Layer {
    mha: MultiHeadAttention,
    pwff: PositionWiseFeedForward,
    norm_1: LayerNorm,
    norm_2: LayerNorm,
}

impl Layer {
    pub fn new(
        mha: MultiHeadAttention,
        pwff: PositionWiseFeedForward,
        norm_1: LayerNorm,
        norm_2: LayerNorm,
    ) -> Self {
        Self {
            mha,
            pwff,
            norm_1,
            norm_2,
        }
    }
}

pub struct LayerState {
    keys: Vec<i32>,
    values: Vec<i32>,
}

pub struct MultiHeadAttention {
    heads: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
}

impl MultiHeadAttention {
    pub fn new(heads: usize, query: Linear, key: Linear, value: Linear, out: Linear) -> Self {
        Self {
            heads,
            query,
            key,
            value,
            out,
        }
    }
}

pub struct PositionWiseFeedForward {
    inner: Linear,
    outer: Linear,
}

impl PositionWiseFeedForward {
    pub fn new(inner: Linear, outer: Linear) -> Self {
        Self { inner, outer }
    }

    pub fn forward_int(&self, input: &[i32]) -> Vec<i32> {
        let mut x = self.inner.forward_int(&input);
        for x in x.iter_mut() {
            *x = (*x).max(0);
        }
        let x = self.outer.forward_int(&x);
        x
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    scale: Vec<i32>,
    bias: Vec<i32>,
}

impl LayerNorm {
    pub fn new(scale: Vec<f32>, bias: Vec<f32>) -> Self {
        Self {
            scale: scale
                .into_iter()
                .map(|x| (x * 127.0).round() as i32)
                .collect(),
            bias: bias
                .into_iter()
                .map(|x| (x * 127.0).round() as i32)
                .collect(),
        }
    }

    // input is scaled by 127
    pub fn forward_int(&self, input: &mut [i32]) {
        let mean = input.iter().sum::<i32>() / input.len() as i32;
        let var = input.iter().map(|x| (x - mean).pow(2)).sum::<i32>() / input.len() as i32;
        let scale = ((var as f32).sqrt() as i32).max(1);
        for i in 0..input.len() {
            input[i] = (input[i] - mean) * self.scale[i] / scale + self.bias[i];
        }
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    #[allow(dead_code)]
    d_input: usize,
    d_output: usize,
    weights_p: Vec<(usize, usize)>,
    weights_n: Vec<(usize, usize)>,
    norm: Option<LayerNorm>,
}

impl Linear {
    pub fn new(d_input: usize, d_output: usize) -> Self {
        Linear {
            d_input,
            d_output,
            weights_p: vec![],
            weights_n: vec![],
            norm: None,
        }
    }

    pub fn from_weights(
        d_input: usize,
        d_output: usize,
        weights: &Vec<f32>,
        _scale: f32,
        norm: Option<LayerNorm>,
    ) -> Self {
        let mut weights_p = vec![];
        let mut weights_n = vec![];
        for i in 0..d_input * d_output {
            if weights[i] > 0.0 {
                weights_p.push((i / d_output, i % d_output));
            } else if weights[i] < 0.0 {
                weights_n.push((i / d_output, i % d_output));
            }
        }
        Linear {
            d_input,
            d_output,
            weights_p,
            weights_n,
            norm,
        }
    }

    pub fn eye(d_input: usize, d_output: usize) -> Self {
        let mut weights_p = vec![];
        for i in 0..d_input.min(d_output) {
            weights_p.push((i, i));
        }
        Linear {
            d_input,
            d_output,
            weights_p,
            weights_n: vec![],
            norm: None,
        }
    }

    pub fn forward_int(&self, input: &[i32]) -> Vec<i32> {
        let mut input = input.to_vec();
        if let Some(norm) = &self.norm {
            norm.forward_int(&mut input)
        }

        let mut output = vec![0; self.d_output];
        for (i, j) in &self.weights_p {
            output[*j] += input[*i];
        }
        for (i, j) in &self.weights_n {
            output[*j] -= input[*i];
        }
        let max = output.iter().map(|x| x.abs()).max().unwrap();
        let output = output.into_iter().map(|x| x * 127 / max).collect();
        output
    }
}

fn softmax_(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter().map(|x| (x - max).exp()).sum();
    for x in x.iter_mut() {
        *x = (*x - max).exp() / sum;
    }
}

// #[test]
// fn test() {
//     let vocab_size = 2;
//     let d_model = 3;
//     let model = BitNet {
//         embedding: vec![vec![0., 1., 0.], vec![1., 0., 1.]],
//         pos: vec![0., 0., 0.].repeat(100),
//         layers: vec![
//             Layer {
//                 mha: MultiHeadAttention {
//                     heads: 2,
//                     query: Linear::eye(d_model, d_model * 2),
//                     key: Linear::eye(d_model, d_model * 2),
//                     value: Linear::eye(d_model, d_model * 2),
//                     out: Linear::eye(d_model * 2, d_model),
//                 },
//                 pwff: PositionWiseFeedForward {
//                     inner: Linear::eye(d_model, d_model),
//                     outer: Linear::eye(d_model, d_model),
//                 },
//                 norm_1: LayerNorm {
//                     scale: vec![1.0, 1.0, 1.0],
//                     bias: vec![0.0, 0.0, 0.0],
//                 },
//                 norm_2: LayerNorm {
//                     scale: vec![1.0, 1.0, 1.0],
//                     bias: vec![0.0, 0.0, 0.0],
//                 },
//             },
//             Layer {
//                 mha: MultiHeadAttention {
//                     heads: 2,
//                     query: Linear::eye(d_model, d_model * 2),
//                     key: Linear::eye(d_model, d_model * 2),
//                     value: Linear::eye(d_model, d_model * 2),
//                     out: Linear::eye(d_model * 2, d_model),
//                 },
//                 pwff: PositionWiseFeedForward {
//                     inner: Linear::eye(d_model, d_model),
//                     outer: Linear::eye(d_model, d_model),
//                 },
//                 norm_1: LayerNorm {
//                     scale: vec![1.0, 1.0, 1.0],
//                     bias: vec![0.0, 0.0, 0.0],
//                 },
//                 norm_2: LayerNorm {
//                     scale: vec![1.0, 1.0, 1.0],
//                     bias: vec![0.0, 0.0, 0.0],
//                 },
//             },
//         ],
//         output: Linear::eye(d_model, vocab_size),
//     };

//     let mut states = BitNetState::vec_from_tokens(&[0, 1, 0, 1]);
//     let logits = BitNetState::vec_compute(&mut states, &model, &[0, 1, 0, 1]);
//     println!("{:?}", logits);
// }
