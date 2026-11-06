//! Will it fit? Predicts where a model's memory goes for a set of run settings: weights on the
//! GPU and in RAM, the KV cache, and llama.cpp's working buffers. Mirrors llama.cpp's placement:
//! the last N layers go to the GPU, the head goes too once every layer does, token embeddings stay
//! in RAM, and `--n-cpu-moe` keeps the experts of the first N layers in RAM.

use super::library::Info;
use crate::store::{FlashAttn, RunPreset};
use cubic_gguf::layout::{local_of, part_of};
use cubic_gguf::Part;

#[derive(Clone, Debug, Default)]
pub struct Estimate {
    pub gpu_layers: u32,
    pub cpu_moe_layers: u32,
    pub weights_gpu: u64,
    pub weights_cpu: u64,
    pub kv_gpu: u64,
    pub kv_cpu: u64,
    pub compute: u64,
    pub overhead: u64,
    pub n_layers: u32,
}

impl Estimate {
    pub fn gpu_total(&self) -> u64 {
        self.weights_gpu + self.kv_gpu + self.compute + self.overhead
    }
}

/// Bytes per KV element for a cache type.
pub fn kv_bytes(t: &str) -> f64 {
    match t {
        "f32" => 4.0,
        "f16" | "bf16" => 2.0,
        "q8_0" => 34.0 / 32.0,
        "q5_1" => 24.0 / 32.0,
        "q5_0" => 22.0 / 32.0,
        "q4_1" => 20.0 / 32.0,
        "q4_0" | "iq4_nl" => 18.0 / 32.0,
        _ => 2.0,
    }
}

pub const KV_TYPES: [&str; 5] = ["f16", "q8_0", "q5_1", "q4_0", "f32"];

/// Driver/context memory the backend takes before any model data.
const OVERHEAD: u64 = 400 << 20;

pub fn estimate(info: &Info, p: &RunPreset, gpu_layers: u32, cpu_moe: u32, has_gpu: bool) -> Estimate {
    let a = &info.anatomy;
    let g = &info.gguf;
    let n_layers = a.layers.len() as u32;
    let gpu_layers = if has_gpu { gpu_layers.min(n_layers + 1) } else { 0 };
    // llama.cpp offloads the last `gpu_layers` repeating layers; the head counts as one more.
    let first_gpu = n_layers.saturating_sub(gpu_layers.min(n_layers));
    let head_on_gpu = gpu_layers > n_layers;
    let mut e = Estimate { gpu_layers, cpu_moe_layers: cpu_moe, n_layers, overhead: if has_gpu { OVERHEAD } else { 0 }, ..Default::default() };

    for (li, l) in a.layers.iter().enumerate() {
        let li = li as u32;
        for &ti in &l.tensors {
            let t = &g.tensors[ti];
            let expert = part_of(local_of(&t.name)) == Part::Experts;
            let on_gpu = li >= first_gpu && !(expert && li < cpu_moe);
            if on_gpu {
                e.weights_gpu += t.n_bytes;
            } else {
                e.weights_cpu += t.n_bytes;
            }
        }
    }
    for &ti in &a.after {
        let t = &g.tensors[ti];
        if head_on_gpu {
            e.weights_gpu += t.n_bytes;
        } else {
            e.weights_cpu += t.n_bytes;
        }
    }
    // Embeddings are looked up row by row on the CPU, so they stay in RAM.
    for &ti in a.before.iter().chain(&a.extra) {
        e.weights_cpu += g.tensors[ti].n_bytes;
    }

    let ctx = p.ctx.max(256) as u64;
    let ub = p.ubatch.max(64) as u64;
    let per = kv_bytes(&p.kv_type);
    for kv in &info.kv {
        let cells = match kv.window {
            Some(w) => (w + ub).min(ctx),
            None => ctx,
        };
        let bytes = (cells as f64 * kv.per_token as f64 * per) as u64;
        if has_gpu && kv.layer as u32 >= first_gpu {
            e.kv_gpu += bytes;
        } else {
            e.kv_cpu += bytes;
        }
    }

    if has_gpu && gpu_layers > 0 {
        let h = &a.hyper;
        let width = h.ffn.max(h.expert_ffn * h.experts_used.max(1)).max(h.embd * 4);
        let mut compute = ub * width * 4 * 3;
        if head_on_gpu {
            compute += ub * h.vocab * 4;
        }
        let fa_on = p.flash_attn != FlashAttn::Off;
        if !fa_on {
            // Without flash attention the full score matrix for a micro-batch is materialised.
            compute += h.heads.max(1) * ctx * ub * 4;
        }
        e.compute = compute;
    }
    e
}

/// The most GPU layers that keep the GPU total under `budget`.
pub fn max_layers(info: &Info, p: &RunPreset, cpu_moe: u32, budget: u64) -> u32 {
    let n = info.anatomy.layers.len() as u32 + 1;
    let mut best = 0;
    for l in 0..=n {
        if estimate(info, p, l, cpu_moe, true).gpu_total() <= budget {
            best = l;
        } else {
            break;
        }
    }
    best
}

/// For a MoE model that doesn't fit: the fewest layers whose experts must stay in RAM so that
/// everything else fits on the GPU.
pub fn min_cpu_moe(info: &Info, p: &RunPreset, budget: u64) -> Option<u32> {
    if info.anatomy.hyper.experts == 0 {
        return None;
    }
    let n = info.anatomy.layers.len() as u32;
    (0..=n).find(|&m| estimate(info, p, n + 1, m, true).gpu_total() <= budget)
}

/// What llama.cpp's own fitting would pick, approximately: all on GPU if it fits; for MoE, move
/// experts to RAM first; otherwise as many layers as fit. Leaves `margin` free like `-fit`.
pub fn auto_plan(info: &Info, p: &RunPreset, budget: u64) -> (u32, u32) {
    let n = info.anatomy.layers.len() as u32;
    if let Some(m) = p.cpu_moe_layers {
        return (max_layers(info, p, m, budget), m);
    }
    if estimate(info, p, n + 1, 0, true).gpu_total() <= budget {
        return (n + 1, 0);
    }
    if let Some(m) = min_cpu_moe(info, p, budget) {
        return (n + 1, m);
    }
    (max_layers(info, p, 0, budget), 0)
}
