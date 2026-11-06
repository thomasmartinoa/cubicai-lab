//! A small model written byte by byte, so the parser, the layout, the KV-cache shape and the
//! statistics can be checked against values known exactly.

use cubic_gguf::layout::{kv_layout, describe, part_of, local_of, Part};
use cubic_gguf::stats::{sample_stats, window, vector, overview, Progress};
use cubic_gguf::{Anatomy, GgmlType, Gguf, ParseError, Writer};

const EMBD: u64 = 64;
const LAYERS: usize = 4;

/// A llama-like model: 4 layers, 4 query heads / 2 KV heads of 16, a sliding window on even layers.
fn model() -> Vec<u8> {
    let mut w = Writer::new();
    w.str("general.architecture", "llama")
        .str("general.name", "Tiny Test")
        .u32("llama.block_count", LAYERS as u32)
        .u32("llama.embedding_length", EMBD as u32)
        .u32("llama.feed_forward_length", 128)
        .u32("llama.attention.head_count", 4)
        .u32("llama.attention.head_count_kv", 2)
        .u32("llama.context_length", 4096)
        .u32("llama.attention.sliding_window", 512)
        .bools("llama.attention.sliding_window_pattern", &[true, false, true, false])
        .f32("llama.rope.freq_base", 10000.0)
        .strs("tokenizer.ggml.tokens", &["<s>", "a", "b", "c", "d", "e", "f", "g", "h", "i"]);
    w.tensor_f32("token_embd.weight", &[EMBD, 10], |i| i as f32 * 0.01);
    for l in 0..LAYERS {
        let b = |n: &str| format!("blk.{l}.{n}.weight");
        w.tensor_f32(&b("attn_norm"), &[EMBD], |i| 1.0 + i as f32 / 64.0)
            .tensor_f32(&b("attn_q"), &[EMBD, 64], |i| ((i % 7) as f32 - 3.0) * 0.1)
            .tensor_f32(&b("attn_k"), &[EMBD, 32], |_| 0.5)
            .tensor_f32(&b("attn_v"), &[EMBD, 32], |i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .tensor_f32(&b("attn_output"), &[64, EMBD], |_| 0.0)
            .tensor_f32(&b("ffn_norm"), &[EMBD], |_| 1.0)
            .tensor_f32(&b("ffn_gate"), &[EMBD, 128], |_| 0.25)
            .tensor_f32(&b("ffn_up"), &[EMBD, 128], |_| 0.25);
        // ffn_down as Q8_0: every block has scale 0.5 and quants -16..15 → values -8.0..7.5.
        let mut q = vec![];
        for _ in 0..(128 * EMBD / 32) {
            q.extend(half::f16::from_f32(0.5).to_le_bytes());
            q.extend((0..32).map(|j| (j as i8 - 16) as u8));
        }
        w.tensor_raw(&b("ffn_down"), &[128, EMBD], GgmlType::Q8_0, q);
    }
    w.tensor_f32("output_norm.weight", &[EMBD], |_| 1.0);
    w.tensor_f32("output.weight", &[EMBD, 10], |_| 0.125);
    w.finish()
}

fn parsed() -> (Gguf, Vec<u8>) {
    let bytes = model();
    let g = Gguf::parse(&bytes, Some(bytes.len() as u64)).expect("parse");
    (g, bytes)
}

fn tensor_bytes<'a>(g: &Gguf, bytes: &'a [u8], name: &str) -> (cubic_gguf::TensorInfo, &'a [u8]) {
    let t = g.tensors.iter().find(|t| t.name == name).unwrap().clone();
    let start = g.tensor_start(&t) as usize;
    (t.clone(), &bytes[start..start + t.n_bytes as usize])
}

#[test]
fn header_and_metadata() {
    let (g, _) = parsed();
    assert_eq!(g.version, 3);
    assert_eq!(g.arch(), "llama");
    assert_eq!(g.get_str("general.name"), Some("Tiny Test"));
    assert_eq!(g.arch_u64("block_count"), Some(4));
    assert_eq!(g.arch_f64("rope.freq_base"), Some(10000.0));
    assert_eq!(g.tensors.len(), 3 + LAYERS * 9);
    assert_eq!(g.data_offset % 32, 0);
    let q = g.tensors.iter().find(|t| t.name == "blk.0.ffn_down.weight").unwrap();
    assert_eq!(q.ty, GgmlType::Q8_0);
    assert_eq!(q.n_bytes, 128 * EMBD / 32 * 34);
    assert_eq!(q.shape_text(), "64 × 128");
    assert_eq!((q.row_len(), q.n_rows()), (128, 64));
}

#[test]
fn truncated_and_invalid_headers() {
    let bytes = model();
    match Gguf::parse(&bytes[..200], None) {
        Err(ParseError::Truncated { have }) => assert_eq!(have, 200),
        other => panic!("expected Truncated, got {other:?}"),
    }
    assert!(matches!(Gguf::parse(b"GGML\x03\0\0\0", None), Err(ParseError::Invalid(_))));
    let mut v9 = bytes.clone();
    v9[4] = 9;
    assert!(matches!(Gguf::parse(&v9, None), Err(ParseError::Invalid(_))));
}

#[test]
fn anatomy_layout() {
    let (g, _) = parsed();
    let a = Anatomy::build(&g);
    assert_eq!(a.layers.len(), LAYERS);
    assert_eq!(a.hyper.n_layers, 4);
    assert_eq!((a.hyper.heads, a.hyper.heads_kv, a.hyper.head_dim), (4, 2, 16));
    assert_eq!(a.hyper.vocab, 10);
    assert_eq!(a.hyper.sliding_window, 512);
    assert_eq!(a.before.len(), 1, "token_embd goes first");
    assert_eq!(a.after.len(), 2);
    assert!(g.tensors[a.after[0]].name.contains("norm"), "norm before the head");
    let per_layer = 64 + 64 * 64 + 2 * 64 * 32 + 64 * 64 + 64 + 2 * 64 * 128 + 128 * 64;
    assert_eq!(a.params, 2 * 64 * 10 + 64 + LAYERS as u64 * per_layer);
    assert_eq!(a.active_params(&g), a.params, "dense model: everything is active");
    let l0 = &a.layers[0];
    assert!(l0.has_attention && !l0.recurrent && !l0.has_experts);
    // Ordered by part: the norms together, then attention, then feed-forward.
    let names: Vec<&str> = l0.tensors.iter().map(|&i| local_of(&g.tensors[i].name)).collect();
    assert_eq!(&names[..3], ["attn_norm.weight", "ffn_norm.weight", "attn_q.weight"]);
    assert_eq!(*names.last().unwrap(), "ffn_down.weight");
    assert_eq!(a.type_mix[0].0, GgmlType::F32);
}

#[test]
fn kv_cache_shape_follows_the_window_pattern() {
    let (g, _) = parsed();
    let a = Anatomy::build(&g);
    let kv = kv_layout(&g, &a);
    assert_eq!(kv.len(), 4);
    for k in &kv {
        assert_eq!(k.per_token, 32 + 32, "K and V rows");
        assert_eq!(k.window, if k.layer % 2 == 0 { Some(512) } else { None }, "layer {}", k.layer);
    }
}

#[test]
fn exact_statistics() {
    let (g, bytes) = parsed();
    let (t, data) = tensor_bytes(&g, &bytes, "blk.0.attn_v.weight");
    let st = sample_stats(&t, data, 1000).unwrap();
    assert_eq!(st.count, 64 * 32);
    assert!(st.mean.abs() < 1e-12);
    assert!((st.std - 1.0).abs() < 1e-9);
    assert_eq!((st.min, st.max, st.absmax), (-1.0, 1.0, 1.0));
    assert_eq!(st.hist.iter().sum::<u32>() as u64, st.count);

    let (t, data) = tensor_bytes(&g, &bytes, "blk.2.attn_output.weight");
    assert_eq!(sample_stats(&t, data, 1000).unwrap().zeros, 1.0);

    // Quantised: Q8_0 rows decode to -8.0..7.5 in steps of 0.5.
    let (t, data) = tensor_bytes(&g, &bytes, "blk.1.ffn_down.weight");
    let st = sample_stats(&t, data, 1000).unwrap();
    assert_eq!((st.min, st.max), (-8.0, 7.5));
    assert!((st.mean - (-0.25)).abs() < 1e-9);
}

#[test]
fn windows_vectors_and_overview() {
    let (g, bytes) = parsed();
    let (t, data) = tensor_bytes(&g, &bytes, "token_embd.weight");
    // Row r, column c holds (r*64 + c) * 0.01.
    let win = window(&t, data, 0, 3, 5, 2, 3).unwrap();
    let want: Vec<f32> = [(3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7)].iter().map(|(r, c)| (r * 64 + c) as f32 * 0.01).collect();
    assert_eq!(win, want);
    // Past the last row the window is padded with zeros.
    assert_eq!(window(&t, data, 0, 9, 0, 2, 1).unwrap()[1], 0.0);

    let (t, data) = tensor_bytes(&g, &bytes, "blk.0.attn_norm.weight");
    let v = vector(&t, data, 16).unwrap();
    assert_eq!(v.len(), 16);
    assert_eq!(v[0].0, 1.0);
    assert!((v[15].1 - (1.0 + 63.0 / 64.0)).abs() < 1e-6);

    let (t, data) = tensor_bytes(&g, &bytes, "blk.0.attn_k.weight");
    let o = overview(&t, data, 0, 16, 8, &Progress::default()).unwrap();
    assert_eq!((o.w, o.h), (16, 8));
    assert!(o.rms.iter().all(|&r| (r - 0.5).abs() < 1e-6), "constant 0.5 has RMS 0.5 everywhere");
}

#[test]
fn split_models_merge_with_their_own_offsets() {
    let (mut a, _) = parsed();
    let (b, _) = parsed();
    let n = a.tensors.len();
    a.merge_split(b, 1);
    assert_eq!(a.tensors.len(), 2 * n);
    let second = &a.tensors[n];
    assert_eq!(second.file, 1);
    assert_eq!(a.tensor_start(second), a.part_offsets[1] + second.offset);
}

#[test]
fn type_ids_round_trip() {
    for id in 0..64 {
        let t = GgmlType::from_id(id);
        assert_eq!(t.id(), id);
    }
}

#[test]
fn names_and_roles() {
    assert_eq!(local_of("blk.12.ffn_up_exps.weight"), "ffn_up_exps.weight");
    assert_eq!(part_of("ffn_up_exps.weight"), Part::Experts);
    assert_eq!(part_of("ffn_gate_inp.weight"), Part::Router);
    assert_eq!(part_of("attn_q.weight"), Part::Attention);
    assert_eq!(part_of("attn_norm.weight"), Part::Norm);
    assert!(!describe("blk.0.attn_q.weight").is_empty());
}
