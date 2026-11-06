//! Steering vectors from contrasting prompts: the average difference between the model's hidden
//! states on "positive" and "negative" versions of the same prompt, per layer, written in
//! llama.cpp's control-vector format (`--control-vector-scaled`).

use crate::parse::Gguf;
use crate::write::Writer;

pub struct Directions {
    /// Mean of (positive − negative) after each layer: `[layer][n_embd]`.
    pub layers: Vec<Vec<f32>>,
    /// How cleanly each layer separates the two sides along its direction (Cohen's d).
    pub separation: Vec<f32>,
    /// Every example projected on its layer's direction: `[layer][example]`, for the scatter.
    pub proj_pos: Vec<Vec<f32>>,
    pub proj_neg: Vec<Vec<f32>>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// `pos[i]` and `neg[i]` are hidden states `[layer][n_embd]` of the i-th pair of prompts.
pub fn directions(pos: &[Vec<Vec<f32>>], neg: &[Vec<Vec<f32>>]) -> Directions {
    assert_eq!(pos.len(), neg.len(), "examples come in pairs");
    let n = pos.len().max(1) as f32;
    let n_layers = pos.first().map(Vec::len).unwrap_or(0);
    let mut out = Directions { layers: vec![], separation: vec![], proj_pos: vec![], proj_neg: vec![] };
    for l in 0..n_layers {
        let e = pos[0][l].len();
        let mut d = vec![0f32; e];
        for (p, q) in pos.iter().zip(neg) {
            for ((d, a), b) in d.iter_mut().zip(&p[l]).zip(&q[l]) {
                *d += (a - b) / n;
            }
        }
        let len = norm(&d);
        let unit: Vec<f32> = if len > 0.0 { d.iter().map(|x| x / len).collect() } else { d.clone() };
        let pp: Vec<f32> = pos.iter().map(|p| dot(&p[l], &unit)).collect();
        let qq: Vec<f32> = neg.iter().map(|q| dot(&q[l], &unit)).collect();
        out.separation.push(cohens_d(&pp, &qq));
        out.proj_pos.push(pp);
        out.proj_neg.push(qq);
        out.layers.push(d);
    }
    out
}

/// (mean a − mean b) / pooled standard deviation; large when the two groups don't overlap.
pub fn cohens_d(a: &[f32], b: &[f32]) -> f32 {
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    let (ma, mb) = (mean(a), mean(b));
    let var = |v: &[f32], m: f32| v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (v.len().saturating_sub(1)).max(1) as f32;
    let pooled = ((var(a, ma) + var(b, mb)) / 2.0).sqrt();
    if pooled < 1e-6 {
        return if (ma - mb).abs() < 1e-6 { 0.0 } else { 10.0f32.copysign(ma - mb) };
    }
    ((ma - mb) / pooled).clamp(-10.0, 10.0)
}

/// A sensible default layer range: the window of about a third of the layers (never layer 0,
/// which llama.cpp can't steer) where the concept separates best.
pub fn best_range(separation: &[f32]) -> (usize, usize) {
    let n = separation.len();
    if n < 2 {
        return (1, 1.max(n.saturating_sub(1)));
    }
    let width = (n / 3).max(1).min(n - 1);
    let mut best = (1, width);
    let mut best_sum = f32::NEG_INFINITY;
    for a in 1..=n - width {
        let s: f32 = separation[a..a + width].iter().map(|x| x.max(0.0)).sum();
        if s > best_sum {
            best_sum = s;
            best = (a, a + width - 1);
        }
    }
    best
}

/// A control-vector GGUF: `direction.N` (F32, n_embd) is added after block N (N ≥ 1).
pub fn control_vector(layers: &[Vec<f32>], model_arch: &str) -> Vec<u8> {
    let mut w = Writer::new();
    w.str("general.architecture", "controlvector")
        .str("controlvector.model_hint", model_arch)
        .u32("controlvector.layer_count", layers.len().saturating_sub(1) as u32);
    for (l, d) in layers.iter().enumerate().skip(1) {
        w.tensor_f32(&format!("direction.{l}"), &[d.len() as u64], |i| d[i as usize]);
    }
    w.finish()
}

/// Reads the directions back from a control-vector file: `[layer][n_embd]`, layer 0 empty.
pub fn read_control_vector(bytes: &[u8]) -> Result<Vec<Vec<f32>>, String> {
    let g = Gguf::parse(bytes, Some(bytes.len() as u64)).map_err(|e| e.to_string())?;
    let mut out: Vec<Vec<f32>> = vec![];
    for t in &g.tensors {
        let l: usize = t.name.strip_prefix("direction.").and_then(|n| n.parse().ok()).ok_or_else(|| format!("unexpected tensor {}", t.name))?;
        if t.ty != crate::GgmlType::F32 {
            return Err(format!("{} isn't F32", t.name));
        }
        let start = g.tensor_start(t) as usize;
        let data = bytes.get(start..start + t.n_bytes as usize).ok_or("file is cut short")?;
        if out.len() <= l {
            out.resize(l + 1, vec![]);
        }
        out[l] = data.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    }
    Ok(out)
}
