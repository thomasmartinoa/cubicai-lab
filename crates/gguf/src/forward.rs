//! A reference forward pass on the CPU for llama-family models (Llama, Mistral, Qwen 2/3), straight
//! from the GGUF weights. It is slow next to llama.cpp but shows what llama.cpp doesn't: the
//! residual stream after every layer. That feeds steering vectors and the layer views.
//!
//! Several prompts run as one batch: their shared beginning is computed once, every weight matrix
//! is read once per layer for all tokens, and each prompt attends only to that shared beginning and
//! its own tokens.

use crate::parse::{Gguf, TensorInfo};
use crate::stats::{Progress, RowReader};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub arch: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    /// Rotate pairs (i, i + d/2) (Qwen) rather than neighbours (i, i + 1) (Llama).
    pub neox: bool,
    pub eps: f32,
    pub vocab: usize,
    /// Rope scaling: "none", "linear" or "yarn", with its factor and the original context.
    pub rope_scaling: String,
    pub rope_factor: f32,
    pub rope_orig_ctx: usize,
}

pub const SUPPORTED: [&str; 3] = ["llama", "qwen2", "qwen3"];

pub struct Model<'a> {
    g: &'a Gguf,
    files: Vec<&'a [u8]>,
    pub cfg: Config,
    freq_factors: Option<Vec<f32>>,
}

struct Mat<'a> {
    r: RowReader<'a>,
}

/// Residual stream at the last token of every prompt, after every layer.
pub struct Hidden {
    /// `[prompt][layer][n_embd]`
    pub last: Vec<Vec<Vec<f32>>>,
    /// Next-token logits after the last token of every prompt (when asked for).
    pub logits: Vec<Vec<f32>>,
}

impl<'a> Model<'a> {
    /// `files` are the bytes of every part (one for unsplit models), as `Gguf::tensor_start` counts.
    pub fn new(g: &'a Gguf, files: Vec<&'a [u8]>) -> Result<Model<'a>, String> {
        let arch = g.arch().to_string();
        if !SUPPORTED.contains(&arch.as_str()) {
            return Err(format!("The layer-by-layer engine doesn't know the {arch} architecture yet (it knows {}).", SUPPORTED.join(", ")));
        }
        let u = |k: &str| g.arch_u64(k).map(|v| v as usize);
        let n_embd = u("embedding_length").ok_or("no embedding_length")?;
        let n_head = u("attention.head_count").ok_or("no head_count")?;
        let head_dim = u("attention.key_length").unwrap_or(n_embd / n_head);
        let vocab = g.tensors.iter().find(|t| t.name == "token_embd.weight").map(|t| t.n_rows() as usize).ok_or("no token_embd")?;
        let cfg = Config {
            n_layer: u("block_count").ok_or("no block_count")?,
            n_embd,
            n_head,
            n_head_kv: u("attention.head_count_kv").unwrap_or(n_head),
            head_dim,
            rope_dim: u("rope.dimension_count").unwrap_or(head_dim),
            rope_base: g.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            neox: arch != "llama",
            eps: g.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            vocab,
            rope_scaling: g.get_str(&format!("{arch}.rope.scaling.type")).unwrap_or("none").to_string(),
            rope_factor: g.arch_f64("rope.scaling.factor").filter(|f| *f > 0.0).unwrap_or(1.0) as f32,
            rope_orig_ctx: u("rope.scaling.original_context_length").unwrap_or(0),
            arch,
        };
        let mut m = Model { g, files, cfg, freq_factors: None };
        m.freq_factors = m.vector("rope_freqs.weight")?;
        // Everything the pass needs must be there and decodable.
        for l in 0..m.cfg.n_layer {
            for n in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"] {
                m.mat(&format!("blk.{l}.{n}.weight"))?;
            }
        }
        m.mat("token_embd.weight")?;
        Ok(m)
    }

    fn tensor(&self, name: &str) -> Option<(&'a TensorInfo, &'a [u8])> {
        let t = self.g.tensors.iter().find(|t| t.name == name)?;
        let file = self.files.get(t.file)?;
        let start = self.g.tensor_start(t) as usize;
        Some((t, file.get(start..start + t.n_bytes as usize)?))
    }

    fn mat(&self, name: &str) -> Result<Mat<'a>, String> {
        let (t, data) = self.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let r = RowReader::new(t, data).ok_or_else(|| format!("{name} is stored as {}, which can't be decoded yet", t.ty.name()))?;
        Ok(Mat { r })
    }

    fn vector(&self, name: &str) -> Result<Option<Vec<f32>>, String> {
        let Some((t, data)) = self.tensor(name) else { return Ok(None) };
        let r = RowReader::new(t, data).ok_or_else(|| format!("{name} can't be decoded"))?;
        let mut v = vec![0f32; r.row_len];
        r.read(0, &mut v);
        Ok(Some(v))
    }

    /// Runs every prompt (token ids) through the model. `want_logits` adds the next-token logits.
    pub fn run(&self, prompts: &[Vec<u32>], want_logits: bool, progress: &Progress) -> Result<Hidden, String> {
        self.run_steered(prompts, want_logits, None, progress)
    }

    /// Like `run`, with a control vector added to every token after each block, the way
    /// llama.cpp applies one: `steer.0[l] * steer.1` after block `l` (empty layers are skipped).
    pub fn run_steered(&self, prompts: &[Vec<u32>], want_logits: bool, steer: Option<(&[Vec<f32>], f32)>, progress: &Progress) -> Result<Hidden, String> {
        let c = &self.cfg;
        if prompts.is_empty() || prompts.iter().any(|p| p.is_empty()) {
            return Err("every prompt needs at least one token".into());
        }
        if let Some(&bad) = prompts.iter().flatten().find(|&&t| t as usize >= c.vocab) {
            return Err(format!("token {bad} is outside the vocabulary"));
        }
        // Shared beginning, leaving each prompt at least one token of its own.
        let min_len = prompts.iter().map(Vec::len).min().unwrap_or(0);
        let mut shared = 0;
        while shared + 1 < min_len && prompts.iter().all(|p| p[shared] == prompts[0][shared]) {
            shared += 1;
        }
        // Flatten: shared tokens (segment 0), then each prompt's own tokens (segment s + 1).
        let mut ids = prompts[0][..shared].to_vec();
        let mut pos: Vec<usize> = (0..shared).collect();
        let mut seg = vec![0usize; shared];
        let mut seg_start = vec![0usize];
        let mut last = vec![];
        for (s, p) in prompts.iter().enumerate() {
            seg_start.push(ids.len());
            for (i, &t) in p.iter().enumerate().skip(shared) {
                ids.push(t);
                pos.push(i);
                seg.push(s + 1);
            }
            last.push(ids.len() - 1);
        }
        let n = ids.len();
        let (e, hd) = (c.n_embd, c.head_dim);
        let (q_dim, kv_dim) = (c.n_head * hd, c.n_head_kv * hd);

        let mut x = vec![0f32; n * e];
        let emb = self.mat("token_embd.weight")?;
        for (i, &t) in ids.iter().enumerate() {
            emb.r.read(t as u64, &mut x[i * e..(i + 1) * e]);
        }
        let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(c.n_layer); prompts.len()];
        let rope = Rope::new(c, self.freq_factors.as_deref());
        let total_steps = c.n_layer + usize::from(want_logits);

        let mut xn = vec![0f32; n * e];
        for l in 0..c.n_layer {
            if progress.cancel.load(Ordering::Relaxed) {
                return Err("Stopped".into());
            }
            let name = |s: &str| format!("blk.{l}.{s}");
            // Attention.
            rms_norm(&x, &mut xn, e, &self.vector(&name("attn_norm.weight"))?.ok_or("no attn_norm")?, c.eps);
            let mut q = matmul(&self.mat(&name("attn_q.weight"))?, &xn, n);
            let mut k = matmul(&self.mat(&name("attn_k.weight"))?, &xn, n);
            let v0 = matmul(&self.mat(&name("attn_v.weight"))?, &xn, n);
            let mut v = v0;
            for (buf, b) in [(&mut q, "attn_q.bias"), (&mut k, "attn_k.bias"), (&mut v, "attn_v.bias")] {
                if let Some(bias) = self.vector(&name(b))? {
                    for row in buf.chunks_exact_mut(bias.len()) {
                        row.iter_mut().zip(&bias).for_each(|(a, b)| *a += b);
                    }
                }
            }
            for (buf, nm) in [(&mut q, "attn_q_norm.weight"), (&mut k, "attn_k_norm.weight")] {
                if let Some(w) = self.vector(&name(nm))? {
                    for head in buf.chunks_exact_mut(hd) {
                        let src = head.to_vec();
                        rms_norm(&src, head, hd, &w, c.eps);
                    }
                }
            }
            for i in 0..n {
                rope.apply(&mut q[i * q_dim..(i + 1) * q_dim], pos[i]);
                rope.apply(&mut k[i * kv_dim..(i + 1) * kv_dim], pos[i]);
            }
            let att = attention(c, &q, &k, &v, &seg, &seg_start, shared);
            let o = matmul(&self.mat(&name("attn_output.weight"))?, &att, n);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);
            // Feed-forward.
            rms_norm(&x, &mut xn, e, &self.vector(&name("ffn_norm.weight"))?.ok_or("no ffn_norm")?, c.eps);
            let mut gate = matmul(&self.mat(&name("ffn_gate.weight"))?, &xn, n);
            let up = matmul(&self.mat(&name("ffn_up.weight"))?, &xn, n);
            gate.iter_mut().zip(&up).for_each(|(g, u)| *g = *g / (1.0 + (-*g).exp()) * u);
            let down = matmul(&self.mat(&name("ffn_down.weight"))?, &gate, n);
            x.iter_mut().zip(&down).for_each(|(a, b)| *a += b);

            if let Some((dirs, scale)) = steer
                && let Some(d) = dirs.get(l).filter(|d| d.len() == e)
            {
                for row in x.chunks_exact_mut(e) {
                    row.iter_mut().zip(d).for_each(|(a, b)| *a += scale * b);
                }
            }
            for (s, &i) in last.iter().enumerate() {
                out[s].push(x[i * e..(i + 1) * e].to_vec());
            }
            progress.done.store(((l + 1) * 1000 / total_steps) as u32, Ordering::Relaxed);
        }

        let mut logits = vec![];
        if want_logits {
            let rows: Vec<f32> = last.iter().flat_map(|&i| x[i * e..(i + 1) * e].iter().copied()).collect();
            let mut normed = vec![0f32; rows.len()];
            let w = self.vector("output_norm.weight")?.ok_or("no output_norm")?;
            rms_norm(&rows, &mut normed, e, &w, c.eps);
            let head = self.mat("output.weight").or_else(|_| self.mat("token_embd.weight"))?;
            let all = matmul(&head, &normed, last.len());
            let vocab = head.r.rows as usize;
            logits = all.chunks_exact(vocab).map(<[f32]>::to_vec).collect();
            progress.done.store(1000, Ordering::Relaxed);
        }
        Ok(Hidden { last: out, logits })
    }
}

pub fn rms_norm(x: &[f32], out: &mut [f32], width: usize, w: &[f32], eps: f32) {
    for (row, o) in x.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let s = 1.0 / (ms + eps).sqrt();
        for ((o, v), w) in o.iter_mut().zip(row).zip(w) {
            *o = v * s * w;
        }
    }
}

struct Rope {
    /// Inverse frequency per rotated pair.
    inv: Vec<f32>,
    head_dim: usize,
    neox: bool,
    /// 1 / scaling factor (1 = no scaling).
    freq_scale: f32,
    /// YaRN: how much each pair keeps its original frequency (0..1), and the cos/sin boost.
    ramp: Vec<f32>,
    mscale: f32,
}

impl Rope {
    fn new(c: &Config, factors: Option<&[f32]>) -> Rope {
        let pairs = c.rope_dim / 2;
        let inv = (0..pairs)
            .map(|i| {
                let f = c.rope_base.powf(-2.0 * i as f32 / c.rope_dim as f32);
                f / factors.and_then(|ff| ff.get(i)).copied().unwrap_or(1.0)
            })
            .collect();
        let mut rope = Rope { inv, head_dim: c.head_dim, neox: c.neox, freq_scale: 1.0, ramp: vec![0.0; pairs], mscale: 1.0 };
        match c.rope_scaling.as_str() {
            "linear" => rope.freq_scale = 1.0 / c.rope_factor,
            "yarn" if c.rope_factor > 1.0 => {
                // ggml's rope_yarn with beta_fast 32, beta_slow 1, ext_factor 1, attn_factor 1.
                rope.freq_scale = 1.0 / c.rope_factor;
                let orig = if c.rope_orig_ctx > 0 { c.rope_orig_ctx } else { 4096 } as f32;
                let dim = |beta: f32| c.rope_dim as f32 * (orig / (beta * 2.0 * std::f32::consts::PI)).ln() / (2.0 * c.rope_base.ln());
                let low = dim(32.0).floor().max(0.0);
                let high = dim(1.0).ceil().min(c.rope_dim as f32 - 1.0);
                rope.ramp = (0..pairs).map(|i| 1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0)).collect();
                rope.mscale = 1.0 + 0.1 * c.rope_factor.ln();
            }
            _ => {}
        }
        rope
    }

    fn apply(&self, heads: &mut [f32], pos: usize) {
        let pairs = self.inv.len();
        for h in heads.chunks_exact_mut(self.head_dim) {
            for (i, inv) in self.inv.iter().enumerate() {
                let extrap = pos as f32 * inv;
                let interp = extrap * self.freq_scale;
                let theta = interp * (1.0 - self.ramp[i]) + extrap * self.ramp[i];
                let (sin, cos) = theta.sin_cos();
                let (sin, cos) = (sin * self.mscale, cos * self.mscale);
                let (a, b) = if self.neox { (i, i + pairs) } else { (2 * i, 2 * i + 1) };
                let (x0, x1) = (h[a], h[b]);
                h[a] = x0 * cos - x1 * sin;
                h[b] = x0 * sin + x1 * cos;
            }
        }
    }
}

/// Causal attention where segment-0 tokens are shared by every prompt and each other segment
/// sees segment 0 plus itself.
fn attention(c: &Config, q: &[f32], k: &[f32], v: &[f32], seg: &[usize], seg_start: &[usize], shared: usize) -> Vec<f32> {
    let n = seg.len();
    let hd = c.head_dim;
    let (q_dim, kv_dim) = (c.n_head * hd, c.n_head_kv * hd);
    let group = c.n_head / c.n_head_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; n * q_dim];
    let threads = threads();
    let chunk = n.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ci, outc) in out.chunks_mut(chunk * q_dim).enumerate() {
            sc.spawn(move || {
                let mut scores = vec![];
                for (li, o) in outc.chunks_exact_mut(q_dim).enumerate() {
                    let i = ci * chunk + li;
                    // Keys this token may look at.
                    let keys: Vec<usize> = if seg[i] == 0 {
                        (0..=i).collect()
                    } else {
                        (0..shared).chain(seg_start[seg[i]]..=i).collect()
                    };
                    for h in 0..c.n_head {
                        let qh = &q[i * q_dim + h * hd..i * q_dim + (h + 1) * hd];
                        let kvh = h / group;
                        scores.clear();
                        let mut max = f32::NEG_INFINITY;
                        for &j in &keys {
                            let kh = &k[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                            let s = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                            max = max.max(s);
                            scores.push(s);
                        }
                        let mut sum = 0.0;
                        for s in scores.iter_mut() {
                            *s = (*s - max).exp();
                            sum += *s;
                        }
                        let oh = &mut o[h * hd..(h + 1) * hd];
                        oh.fill(0.0);
                        for (&j, s) in keys.iter().zip(&scores) {
                            let w = s / sum;
                            let vh = &v[j * kv_dim + kvh * hd..j * kv_dim + (kvh + 1) * hd];
                            oh.iter_mut().zip(vh).for_each(|(o, v)| *o += w * v);
                        }
                    }
                }
            });
        }
    });
    out
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

struct SharedOut(*mut f32);
unsafe impl Send for SharedOut {}
unsafe impl Sync for SharedOut {}

const ROW_BLOCK: usize = 32;

/// `x` (n × cols) times the matrix's rows → n × rows. Rows are decoded once per call, in blocks
/// spread over all cores.
fn matmul(m: &Mat, x: &[f32], n: usize) -> Vec<f32> {
    let rows = m.r.rows as usize;
    let cols = m.r.row_len;
    debug_assert_eq!(x.len(), n * cols);
    let mut out = vec![0f32; n * rows];
    let ptr = SharedOut(out.as_mut_ptr());
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..threads().min(rows.div_ceil(ROW_BLOCK)) {
            let (ptr, next) = (&ptr, &next);
            sc.spawn(move || {
                let mut w = vec![0f32; ROW_BLOCK * cols];
                let mut res = vec![0f32; ROW_BLOCK * n];
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    let r0 = b * ROW_BLOCK;
                    if r0 >= rows {
                        break;
                    }
                    let nr = ROW_BLOCK.min(rows - r0);
                    for r in 0..nr {
                        m.r.read((r0 + r) as u64, &mut w[r * cols..(r + 1) * cols]);
                    }
                    block(&w[..nr * cols], cols, x, n, &mut res[..nr * n]);
                    for r in 0..nr {
                        for t in 0..n {
                            // Disjoint: each (t, row) is written by exactly one block.
                            unsafe { *ptr.0.add(t * rows + r0 + r) = res[r * n + t] };
                        }
                    }
                }
            });
        }
    });
    out
}

/// res[r][t] = w[r] · x[t]
fn block(w: &[f32], cols: usize, x: &[f32], n: usize, res: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { block_avx2(w, cols, x, n, res) };
        }
    }
    block_plain(w, cols, x, n, res)
}

/// Tokens handled together while a block of weight rows sits in cache.
const TOK_BLOCK: usize = 16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn block_avx2(w: &[f32], cols: usize, x: &[f32], n: usize, res: &mut [f32]) {
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(_mm_hadd_ps(s, s))
    }
    let nr = w.len() / cols;
    let body = cols - cols % 8;
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    let tail = |r: usize, t: usize, s: f32| -> f32 { s + (body..cols).map(|j| w[r * cols + j] * x[t * cols + j]).sum::<f32>() };
    for t0 in (0..n).step_by(TOK_BLOCK) {
        let t1 = (t0 + TOK_BLOCK).min(n);
        let mut r = 0;
        while r < nr {
            let two = r + 1 < nr;
            let (w0, w1) = (wp.add(r * cols), wp.add((r + usize::from(two)) * cols));
            let mut t = t0;
            while t < t1 {
                let k = (t1 - t).min(4);
                // 2 rows × up to 4 tokens: every weight load feeds 4 FMAs, every token load 2.
                let xs: [*const f32; 4] = std::array::from_fn(|i| xp.add((t + i.min(k - 1)) * cols));
                let mut acc = [_mm256_setzero_ps(); 8];
                let mut j = 0;
                while j < body {
                    let a = _mm256_loadu_ps(w0.add(j));
                    let b = _mm256_loadu_ps(w1.add(j));
                    for i in 0..4 {
                        let xv = _mm256_loadu_ps(xs[i].add(j));
                        acc[i] = _mm256_fmadd_ps(a, xv, acc[i]);
                        acc[4 + i] = _mm256_fmadd_ps(b, xv, acc[4 + i]);
                    }
                    j += 8;
                }
                for i in 0..k {
                    res[r * n + t + i] = tail(r, t + i, hsum(acc[i]));
                    if two {
                        res[(r + 1) * n + t + i] = tail(r + 1, t + i, hsum(acc[4 + i]));
                    }
                }
                t += 4;
            }
            r += 2;
        }
    }
}

fn block_plain(w: &[f32], cols: usize, x: &[f32], n: usize, res: &mut [f32]) {
    for (r, wr) in w.chunks_exact(cols).enumerate() {
        for t in 0..n {
            let xr = &x[t * cols..(t + 1) * cols];
            let mut acc = [0f32; 8];
            let (wa, wt) = wr.split_at(cols - cols % 8);
            let (xa, xt) = xr.split_at(cols - cols % 8);
            for (a, b) in wa.as_chunks::<8>().0.iter().zip(xa.as_chunks::<8>().0) {
                for j in 0..8 {
                    acc[j] += a[j] * b[j];
                }
            }
            let mut s: f32 = acc.iter().sum();
            for (a, b) in wt.iter().zip(xt) {
                s += a * b;
            }
            res[r * n + t] = s;
        }
    }
}

/// Probabilities from logits (softmax), most likely first: (token id, probability).
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = k.min(idx.len());
    idx.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| logits[b].total_cmp(&logits[a]));
    let mut top: Vec<(usize, f32)> = idx[..k].iter().map(|&i| (i, (logits[i] - max).exp() / sum)).collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top
}
