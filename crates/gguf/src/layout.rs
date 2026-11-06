//! A model's anatomy: its hyperparameters and tensors grouped into layers and parts, with
//! plain-language roles. Built only from the file, nothing runs.

use crate::parse::{Gguf, TensorInfo};
use crate::types::GgmlType;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Part {
    Norm,
    Attention,
    Router,
    Experts,
    SharedExpert,
    Ffn,
    Other,
}

impl Part {
    pub fn label(self) -> &'static str {
        match self {
            Part::Norm => "Norms",
            Part::Attention => "Attention",
            Part::Router => "Router",
            Part::Experts => "Experts",
            Part::SharedExpert => "Shared expert",
            Part::Ffn => "Feed-forward",
            Part::Other => "Other",
        }
    }
}

/// Which part of a block a tensor belongs to, from its name after `blk.N.`.
pub fn part_of(local: &str) -> Part {
    let base = base_name(local);
    if base.ends_with("_norm") || base.starts_with("norm") || base.contains("norm") && !base.contains("ffn_gate") {
        return Part::Norm;
    }
    if base == "ffn_gate_inp" || base == "ffn_gate_inp_shexp" || base.starts_with("exp_probs") {
        return Part::Router;
    }
    if base.ends_with("_exps") {
        return Part::Experts;
    }
    if base.ends_with("_shexp") {
        return Part::SharedExpert;
    }
    if base.starts_with("attn") || base.starts_with("wq") || base.starts_with("wk") || base.starts_with("wv") {
        return Part::Attention;
    }
    if base.starts_with("ffn") {
        return Part::Ffn;
    }
    Part::Other
}

/// `attn_q.weight` → `attn_q`.
pub fn base_name(local: &str) -> &str {
    for suffix in [".weight", ".bias", ".scale", ".scales"] {
        if let Some(b) = local.strip_suffix(suffix) {
            return b;
        }
    }
    local
}

/// What a tensor does, in a sentence anyone can follow.
pub fn describe(name: &str) -> &'static str {
    let local = name.split_once('.').filter(|(b, _)| *b == "blk").map(|(_, r)| r.split_once('.').map(|x| x.1).unwrap_or(r));
    let base = base_name(local.unwrap_or(name));
    if name.ends_with(".bias") {
        return "A bias: a constant added after the matrix above it";
    }
    match base {
        "token_embd" => "Token embeddings: one learned vector per vocabulary entry, the model's starting point for every token",
        "token_embd_norm" => "Normalises the embeddings before the first layer",
        "output_norm" => "Final normalisation before the prediction head",
        "output" => "Prediction head: turns the last hidden state into a score for every token in the vocabulary",
        "rope_freqs" => "Rotary position frequencies: how token positions are encoded",
        "attn_norm" => "Normalises the hidden state before attention",
        "attn_q" => "Query projection: each token turns into the question it asks the others",
        "attn_k" => "Key projection: what each token offers to be found by",
        "attn_v" => "Value projection: the information a token hands over when attended to",
        "attn_qkv" => "Query, key and value projections fused into one matrix",
        "attn_output" | "attn_out" => "Attention output: mixes what all heads found back into the hidden state",
        "attn_q_norm" | "attn_k_norm" => "Per-head normalisation of queries/keys, keeps attention scores stable",
        "attn_gate" => "Attention output gate: lets each head's result through or holds it back",
        "post_norm" => "Normalises the layer's output",
        "inp_gate" | "proj" => "Per-layer embedding path: mixes a per-layer token signal into this layer",
        "attn_sinks" => "Attention sinks: a learned place for heads to look when nothing matters",
        "attn_post_norm" | "post_attention_norm" => "Normalises attention's output before it joins the residual stream",
        "ffn_norm" | "pre_ffw_norm" => "Normalises the hidden state before the feed-forward block",
        "post_ffw_norm" | "ffn_post_norm" => "Normalises the feed-forward output before it joins the residual stream",
        "ffn_gate" => "Feed-forward gate: decides how much of each hidden feature passes (SwiGLU/GeGLU)",
        "ffn_up" => "Feed-forward up projection: expands into a wider space where most knowledge is stored",
        "ffn_down" => "Feed-forward down projection: folds the wide space back into the hidden state",
        "ffn_gate_inp" => "Router: scores every expert for each token; the top few get to run",
        "ffn_gate_exps" => "Experts' gates, one slice per expert",
        "ffn_up_exps" => "Experts' up projections, one slice per expert",
        "ffn_down_exps" => "Experts' down projections, one slice per expert",
        "ffn_gate_shexp" | "ffn_up_shexp" | "ffn_down_shexp" => "Shared expert: runs for every token alongside the routed ones",
        "ffn_gate_inp_shexp" => "Gate for the shared expert's contribution",
        "exp_probs_b" => "Router bias: balances how often each expert is picked",
        "layer_output_scale" => "Scales the whole layer's contribution",
        "ssm_conv1d" => "Short convolution over the last few tokens, feeding the recurrent mixer",
        "ssm_out" => "Recurrent mixer output projection",
        "ssm_norm" => "Normalises the recurrent mixer's output",
        _ if base.starts_with("ssm") => "Linear-attention / state-space mixer: keeps a fixed-size memory instead of a growing KV cache",
        _ if base.contains("altup") || base.contains("laurel") || base.starts_with("per_layer") || base.starts_with("inp_gate") || base.starts_with("proj") => {
            "Per-layer embedding / side-path machinery (Gemma-n style)"
        }
        _ if base.starts_with("v.") || name.starts_with("v.") || name.starts_with("mm.") => "Vision encoder part",
        _ => "",
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub index: usize,
    /// Tensor indices into `Gguf::tensors`, ordered by part.
    pub tensors: Vec<usize>,
    pub bytes: u64,
    pub has_attention: bool,
    pub has_experts: bool,
    /// A state-space / linear-attention layer: fixed-size state, no KV cache that grows with context.
    pub recurrent: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Hyper {
    pub arch: String,
    pub name: String,
    pub size_label: String,
    pub n_layers: u64,
    pub embd: u64,
    pub heads: u64,
    pub heads_kv: u64,
    pub head_dim: u64,
    pub ffn: u64,
    pub experts: u64,
    pub experts_used: u64,
    pub expert_ffn: u64,
    pub ctx_train: u64,
    pub vocab: u64,
    pub rope_base: f64,
    pub sliding_window: u64,
}

#[derive(Clone, Debug)]
pub struct Anatomy {
    pub hyper: Hyper,
    pub params: u64,
    pub bytes: u64,
    pub layers: Vec<Layer>,
    /// Tensors before the first block (embeddings) and after the last (final norm, head), and others.
    pub before: Vec<usize>,
    pub after: Vec<usize>,
    pub extra: Vec<usize>,
    /// Bytes and tensor count per type, largest first.
    pub type_mix: Vec<(GgmlType, u64, usize)>,
}

impl Anatomy {
    pub fn build(g: &Gguf) -> Anatomy {
        let mut layers: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let (mut before, mut after, mut extra) = (vec![], vec![], vec![]);
        for (i, t) in g.tensors.iter().enumerate() {
            if let Some(rest) = t.name.strip_prefix("blk.")
                && let Some((n, _)) = rest.split_once('.')
                    && let Ok(n) = n.parse::<usize>() {
                        layers.entry(n).or_default().push(i);
                        continue;
                    }
            let base = base_name(&t.name);
            if base.starts_with("token_embd") || base == "rope_freqs" || base.starts_with("per_layer") || base.starts_with("altup") {
                before.push(i);
            } else if base.starts_with("output") {
                after.push(i);
            } else {
                extra.push(i);
            }
        }
        // The head after its norm.
        after.sort_by_key(|&i| if g.tensors[i].name.contains("norm") { 0 } else { 1 });

        let layers: Vec<Layer> = layers
            .into_iter()
            .map(|(index, mut tensors)| {
                tensors.sort_by(|&a, &b| {
                    let (ta, tb) = (&g.tensors[a], &g.tensors[b]);
                    let la = local_of(&ta.name);
                    let lb = local_of(&tb.name);
                    (part_of(la), order_in_part(la), la).cmp(&(part_of(lb), order_in_part(lb), lb))
                });
                let bytes = tensors.iter().map(|&i| g.tensors[i].n_bytes).sum();
                let recurrent = tensors.iter().any(|&i| base_name(local_of(&g.tensors[i].name)).starts_with("ssm"));
                let has_attention = !recurrent
                    && tensors.iter().any(|&i| {
                        let b = base_name(local_of(&g.tensors[i].name));
                        matches!(b, "attn_k" | "attn_qkv" | "attn_kv_a_mqa" | "attn_k_b")
                    });
                let has_experts = tensors.iter().any(|&i| part_of(local_of(&g.tensors[i].name)) == Part::Experts);
                Layer { index, tensors, bytes, has_attention, has_experts, recurrent }
            })
            .collect();

        let mut mix: BTreeMap<GgmlType, (u64, usize)> = BTreeMap::new();
        for t in &g.tensors {
            let e = mix.entry(t.ty).or_default();
            e.0 += t.n_bytes;
            e.1 += 1;
        }
        let mut type_mix: Vec<_> = mix.into_iter().map(|(t, (b, c))| (t, b, c)).collect();
        type_mix.sort_by_key(|t| std::cmp::Reverse(t.1));

        let heads = g.arch_u64("attention.head_count").unwrap_or(0);
        let embd = g.arch_u64("embedding_length").unwrap_or(0);
        let hyper = Hyper {
            arch: g.arch().to_string(),
            name: g.get_str("general.name").unwrap_or("").to_string(),
            size_label: g.get_str("general.size_label").unwrap_or("").to_string(),
            n_layers: g.arch_u64("block_count").unwrap_or(layers.len() as u64),
            embd,
            heads,
            heads_kv: g.arch_u64("attention.head_count_kv").unwrap_or(heads),
            head_dim: g
                .arch_u64("attention.key_length")
                .unwrap_or(embd.checked_div(heads).unwrap_or(0)),
            ffn: g.arch_u64("feed_forward_length").unwrap_or(0),
            experts: g.arch_u64("expert_count").unwrap_or(0),
            experts_used: g.arch_u64("expert_used_count").unwrap_or(0),
            expert_ffn: g.arch_u64("expert_feed_forward_length").unwrap_or(0),
            ctx_train: g.arch_u64("context_length").unwrap_or(0),
            vocab: g
                .arch_u64("vocab_size")
                .or_else(|| g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()).map(|a| a.len))
                .unwrap_or(0),
            rope_base: g.arch_f64("rope.freq_base").unwrap_or(0.0),
            sliding_window: g.arch_u64("attention.sliding_window").unwrap_or(0),
        };
        Anatomy {
            hyper,
            params: g.tensors.iter().map(|t| t.n_elements).sum(),
            bytes: g.total_tensor_bytes(),
            layers,
            before,
            after,
            extra,
            type_mix,
        }
    }

    /// Parameters that run for each token: everything except the experts that sit idle.
    pub fn active_params(&self, g: &Gguf) -> u64 {
        let h = &self.hyper;
        if h.experts == 0 || h.experts_used == 0 {
            return self.params;
        }
        let experts: u64 = g
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("blk.") && part_of(local_of(&t.name)) == Part::Experts)
            .map(|t| t.n_elements)
            .sum();
        self.params - experts + experts * h.experts_used / h.experts
    }
}

/// `blk.3.attn_q.weight` → `attn_q.weight`.
pub fn local_of(name: &str) -> &str {
    name.strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map(|(_, l)| l)
        .unwrap_or(name)
}

fn order_in_part(local: &str) -> usize {
    let b = base_name(local);
    const ORDER: [&str; 16] = [
        "attn_norm", "attn_q", "attn_qkv", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_output", "attn_sinks",
        "ffn_norm", "ffn_gate_inp", "ffn_gate", "ffn_up", "ffn_down", "ffn_gate_exps", "ffn_up_exps",
    ];
    ORDER.iter().position(|o| *o == b).unwrap_or(ORDER.len())
}

/// One layer's KV cache need: elements stored per token (K and V together), and the window it is
/// limited to when the layer uses sliding-window attention.
#[derive(Clone, Copy, Debug)]
pub struct KvLayer {
    /// The block's index (`blk.N`).
    pub layer: usize,
    pub per_token: u64,
    pub window: Option<u64>,
}

/// The KV cache each layer keeps, from the tensor shapes (robust across architectures) and the
/// sliding-window / shared-KV metadata. Layers without a cache (recurrent, shared) are left out.
pub fn kv_layout(g: &Gguf, a: &Anatomy) -> Vec<KvLayer> {
    let arch = g.arch();
    let swa = g.arch_u64("attention.sliding_window").unwrap_or(0);
    let pattern: Option<Vec<bool>> = g
        .get(&format!("{arch}.attention.sliding_window_pattern"))
        .and_then(|v| v.as_array())
        .map(|arr| arr.items.iter().map(|v| v.as_bool().unwrap_or(false)).collect());
    // llama.cpp's built-in patterns for architectures that don't store one: layer il slides
    // when il % n < n - 1.
    let n_pattern = match arch {
        "gemma2" | "gptoss" | "gpt-oss" => 2,
        "gemma3" | "gemma3n" => 6,
        "cohere2" => 4,
        _ => 0,
    };
    let shared = g.arch_u64("attention.shared_kv_layers").unwrap_or(0) as usize;
    let n_layers = a.layers.len();
    let tensor = |l: &Layer, base: &str| {
        l.tensors.iter().map(|&i| &g.tensors[i]).find(|t| base_name(local_of(&t.name)) == base)
    };
    let mut out = Vec::new();
    for l in &a.layers {
        if !l.has_attention || (shared > 0 && l.index >= n_layers.saturating_sub(shared)) {
            continue;
        }
        let ne1 = |t: Option<&TensorInfo>| t.and_then(|t| t.dims.get(1).copied());
        let k = ne1(tensor(l, "attn_k"));
        let v = ne1(tensor(l, "attn_v"));
        let per_token = match (k, v) {
            (Some(k), Some(v)) => k + v,
            (Some(k), None) => 2 * k,
            _ => match ne1(tensor(l, "attn_qkv")) {
                Some(qkv) => qkv.saturating_sub(a.hyper.heads * a.hyper.head_dim),
                None => {
                    let h = &a.hyper;
                    2 * h.heads_kv * h.head_dim
                }
            },
        };
        let slides = match &pattern {
            Some(p) => p.get(l.index).copied().unwrap_or(false),
            None => swa > 0 && n_pattern > 0 && l.index % n_pattern < n_pattern - 1,
        };
        out.push(KvLayer { layer: l.index, per_token, window: (slides && swa > 0).then_some(swa) });
    }
    out
}

/// True when the file carries vision/audio towers inside it (Ollama's own packaging), which
/// upstream llama.cpp expects in a separate mmproj file.
pub fn bundles_towers(g: &Gguf) -> bool {
    g.tensors.iter().any(|t| t.name.starts_with("v.") || t.name.starts_with("a.") || t.name.starts_with("mm."))
}
