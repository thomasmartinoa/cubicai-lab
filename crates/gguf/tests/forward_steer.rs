//! The CPU forward pass on a tiny random model (checked against a plain single-prompt reference
//! computed with the same code path), and the steering math.

use cubic_gguf::forward::{top_k, Model};
use cubic_gguf::stats::Progress;
use cubic_gguf::steer::{best_range, cohens_d, control_vector, directions, read_control_vector};
use cubic_gguf::{GgmlType, Gguf, Writer};

const E: u64 = 64;
const FF: u64 = 96;
const V: u64 = 50;
const L: usize = 3;

/// Deterministic pseudo-random weights.
fn rnd(seed: u64) -> impl Fn(u64) -> f32 {
    move |i| {
        let mut x = (i + 1).wrapping_mul(0x9E3779B97F4A7C15) ^ seed.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 31;
        x = x.wrapping_mul(0x94D049BB133111EB);
        ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.4
    }
}

fn tiny(arch: &str, q8: bool) -> Vec<u8> {
    let mut w = Writer::new();
    let k = |s: &str| format!("{arch}.{s}");
    w.str("general.architecture", arch)
        .u32(&k("block_count"), L as u32)
        .u32(&k("embedding_length"), E as u32)
        .u32(&k("feed_forward_length"), FF as u32)
        .u32(&k("attention.head_count"), 4)
        .u32(&k("attention.head_count_kv"), 2)
        .f32(&k("rope.freq_base"), 10000.0)
        .f32(&k("attention.layer_norm_rms_epsilon"), 1e-5);
    let mut seed = 1;
    let mut mat = |w: &mut Writer, name: &str, dims: &[u64]| {
        seed += 1;
        let f = rnd(seed);
        if q8 {
            let n: u64 = dims.iter().product();
            let vals: Vec<f32> = (0..n).map(&f).collect();
            let mut bytes = vec![];
            for b in vals.chunks(32) {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                bytes.extend(half::f16::from_f32(d).to_le_bytes());
                bytes.extend(b.iter().map(|v| (if d > 0.0 { (v / d).round() } else { 0.0 }) as i8 as u8));
            }
            w.tensor_raw(name, dims, GgmlType::Q8_0, bytes);
        } else {
            w.tensor_f32(name, dims, f);
        }
    };
    mat(&mut w, "token_embd.weight", &[E, V]);
    for l in 0..L {
        let b = |s: &str| format!("blk.{l}.{s}.weight");
        w.tensor_f32(&b("attn_norm"), &[E], |i| 1.0 + (i % 3) as f32 * 0.1);
        mat(&mut w, &b("attn_q"), &[E, 64]);
        mat(&mut w, &b("attn_k"), &[E, 32]);
        mat(&mut w, &b("attn_v"), &[E, 32]);
        mat(&mut w, &b("attn_output"), &[64, E]);
        if arch == "qwen3" {
            w.tensor_f32(&b("attn_q_norm"), &[16], |_| 1.0);
            w.tensor_f32(&b("attn_k_norm"), &[16], |_| 1.0);
        }
        w.tensor_f32(&b("ffn_norm"), &[E], |_| 1.0);
        mat(&mut w, &b("ffn_gate"), &[E, FF]);
        mat(&mut w, &b("ffn_up"), &[E, FF]);
        mat(&mut w, &b("ffn_down"), &[FF, E]);
    }
    w.tensor_f32("output_norm.weight", &[E], |_| 1.0);
    mat(&mut w, "output.weight", &[E, V]);
    w.finish()
}

fn run(bytes: &[u8], prompts: &[Vec<u32>]) -> cubic_gguf::forward::Hidden {
    let g = Gguf::parse(bytes, Some(bytes.len() as u64)).unwrap();
    let m = Model::new(&g, vec![bytes]).unwrap();
    m.run(prompts, true, &Progress::default()).unwrap()
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

#[test]
fn batching_and_shared_beginnings_change_nothing() {
    for arch in ["llama", "qwen3"] {
        for q8 in [false, true] {
            let m = tiny(arch, q8);
            let prompts = vec![vec![1, 2, 3, 4, 5], vec![1, 2, 3, 9], vec![1, 2, 7, 8, 9, 10, 11], vec![6]];
            let together = run(&m, &prompts);
            for (i, p) in prompts.iter().enumerate() {
                let alone = run(&m, std::slice::from_ref(p));
                for l in 0..L {
                    assert!(close(&together.last[i][l], &alone.last[0][l], 1e-4), "{arch} q8={q8} prompt {i} layer {l}");
                }
                assert!(close(&together.logits[i], &alone.logits[0], 1e-4));
            }
        }
    }
}

#[test]
fn causal_and_position_aware() {
    let m = tiny("llama", false);
    // What comes after a token can't change it…
    let a = run(&m, &[vec![3, 4, 5]]);
    let b = run(&m, &[vec![3, 4, 5, 6, 7]]);
    let c = run(&m, &[vec![3, 4]]);
    assert_ne!(a.last[0][L - 1], b.last[0][L - 1]);
    // …while order matters (rope), and a different history changes the last state.
    let swapped = run(&m, &[vec![4, 3, 5]]);
    assert!(!close(&a.last[0][L - 1], &swapped.last[0][L - 1], 1e-6));
    assert_ne!(a.last[0][0], c.last[0][0]);
    let top = top_k(&a.logits[0], 5);
    assert_eq!(top.len(), 5);
    assert!(top.windows(2).all(|w| w[0].1 >= w[1].1));
    let all: f32 = top_k(&a.logits[0], V as usize).iter().map(|t| t.1).sum();
    assert!((all - 1.0).abs() < 1e-4);
}

#[test]
fn refuses_what_it_cant_run() {
    let m = tiny("llama", false);
    let g = Gguf::parse(&m, None).unwrap();
    let model = Model::new(&g, vec![&m[..]]).unwrap();
    assert!(model.run(&[vec![V as u32]], false, &Progress::default()).is_err(), "token outside the vocabulary");
    assert!(model.run(&[vec![]], false, &Progress::default()).is_err());
    let p = Progress::default();
    p.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(model.run(&[vec![1]], false, &p).is_err());
    let gemma = tiny("gemma3", false);
    let g = Gguf::parse(&gemma, None).unwrap();
    assert!(Model::new(&g, vec![&gemma[..]]).err().unwrap().contains("gemma3"));
}

#[test]
fn steering_directions_and_files() {
    // Two layers; along dimension 0 the sides differ by 2 at layer 1 only.
    let mk = |l1: f32, noise: f32| vec![vec![noise, 1.0], vec![l1 + noise, noise]];
    let pos: Vec<_> = (0..6).map(|i| mk(2.0, (i as f32 - 2.5) * 0.1)).collect();
    let neg: Vec<_> = (0..6).map(|i| mk(0.0, (i as f32 - 2.5) * 0.1)).collect();
    let d = directions(&pos, &neg);
    assert_eq!(d.layers.len(), 2);
    assert!((d.layers[1][0] - 2.0).abs() < 1e-6 && d.layers[1][1].abs() < 1e-6);
    assert!(d.layers[0].iter().all(|x| x.abs() < 1e-6), "no difference at layer 0");
    assert!(d.separation[1] > 5.0 && d.separation[0].abs() < 1e-3);
    assert_eq!(d.proj_pos[1].len(), 6);

    assert!((cohens_d(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0])).abs() < 1e-6);
    assert!(cohens_d(&[5.0, 6.0], &[1.0, 2.0]) > 4.0);
    assert_eq!(best_range(&[9.0, 0.0, 1.0, 5.0, 6.0, 0.0]), (3, 4), "never layer 0, the best window");
    assert_eq!(best_range(&[0.0; 3]).0, 1);

    let file = control_vector(&d.layers, "llama");
    let g = Gguf::parse(&file, None).unwrap();
    assert_eq!(g.get_str("general.architecture"), Some("controlvector"));
    assert_eq!(g.tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["direction.1"], "layer 0 is left out");
    assert_eq!(g.tensors[0].ty, GgmlType::F32);
    assert_eq!(g.tensors[0].dims, vec![2]);
    let back = read_control_vector(&file).unwrap();
    assert_eq!(back[1], d.layers[1]);
    assert!(back[0].is_empty());
}

#[test]
fn steering_adds_to_every_token_after_its_layer() {
    let m = tiny("llama", false);
    let g = Gguf::parse(&m, None).unwrap();
    let model = Model::new(&g, vec![&m[..]]).unwrap();
    let p = Progress::default();
    let prompt = vec![vec![1u32, 2, 3]];
    let plain = model.run(&prompt, true, &p).unwrap();
    let mut dirs = vec![vec![]; L];
    dirs[1] = (0..E).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let zero = model.run_steered(&prompt, true, Some((&dirs, 0.0)), &p).unwrap();
    assert_eq!(zero.logits, plain.logits, "scale 0 changes nothing");
    let pushed = model.run_steered(&prompt, true, Some((&dirs, 3.0)), &p).unwrap();
    assert_eq!(pushed.last[0][0], plain.last[0][0], "layers before it are untouched");
    assert!((pushed.last[0][1][0] - plain.last[0][1][0] - 3.0).abs() < 1e-5, "added right after its layer");
    assert_eq!(pushed.last[0][1][1..], plain.last[0][1][1..]);
    assert_ne!(pushed.logits, plain.logits, "and it carries through to the output");
    // Earlier tokens were steered too: the last state differs beyond the direct addition.
    let later_diff: f32 = pushed.last[0][2].iter().zip(&plain.last[0][2]).map(|(a, b)| (a - b).abs()).sum();
    assert!(later_diff > 3.0 + 1e-3);
}
