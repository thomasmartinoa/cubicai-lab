//! Parakeet TDT's parts that aren't ONNX: NeMo's log-mel features (onnx-asr's `nemo128`), greedy
//! token-and-duration decoding, and tokens → text. Ports of onnx-asr's numpy code.

use regex::Regex;
use std::sync::LazyLock;

pub const N_FFT: usize = 512;
pub const WIN: usize = 400;
pub const HOP: usize = 160;
pub const MELS: usize = 128;
pub const BINS: usize = N_FFT / 2 + 1;
pub const MAX_TOKENS_PER_STEP: usize = 10;

// --- Mel filterbank (librosa.filters.mel, Slaney scale and norm) ------------------------------------

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;

fn logstep() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel(f: f64) -> f64 {
    let min_log_mel = MIN_LOG_HZ / F_SP;
    if f >= MIN_LOG_HZ { min_log_mel + (f / MIN_LOG_HZ).ln() / logstep() } else { f / F_SP }
}

fn mel_to_hz(m: f64) -> f64 {
    let min_log_mel = MIN_LOG_HZ / F_SP;
    if m >= min_log_mel { MIN_LOG_HZ * (logstep() * (m - min_log_mel)).exp() } else { F_SP * m }
}

/// The 257×128 filterbank (row = FFT bin), as float32 like librosa makes it for 16 kHz.
pub fn mel_bank() -> Vec<f32> {
    let sr = 16000.0;
    let fft_freqs: Vec<f64> = (0..BINS).map(|i| i as f64 * (sr / 2.0) / (BINS - 1) as f64).collect();
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let n = MELS + 2;
    let mel_f: Vec<f64> = (0..n).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n - 1) as f64)).collect();
    let mut bank = vec![0f32; BINS * MELS];
    for m in 0..MELS {
        let (d0, d1) = (mel_f[m + 1] - mel_f[m], mel_f[m + 2] - mel_f[m + 1]);
        let enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
        for (b, &f) in fft_freqs.iter().enumerate() {
            let lower = -(mel_f[m] - f) / d0;
            let upper = (mel_f[m + 2] - f) / d1;
            let w = 0f64.max(lower.min(upper)) as f32;
            bank[b * MELS + m] = (w as f64 * enorm) as f32;
        }
    }
    bank
}

// --- FFT (radix 2, 512 points) -----------------------------------------------------------------------

/// |rfft(x)|² for a 512-sample frame, as numpy gets it: magnitude in float64, then float32, squared.
pub struct Fft {
    cos: Vec<f64>,
    sin: Vec<f64>,
    rev: Vec<usize>,
}

impl Fft {
    pub fn new() -> Fft {
        let bits = N_FFT.trailing_zeros();
        Fft {
            cos: (0..N_FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).cos()).collect(),
            sin: (0..N_FFT / 2).map(|k| (-2.0 * std::f64::consts::PI * k as f64 / N_FFT as f64).sin()).collect(),
            rev: (0..N_FFT).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect(),
        }
    }

    pub fn power(&self, frame: &[f64], out: &mut [f32]) {
        let mut re: Vec<f64> = (0..N_FFT).map(|i| frame[self.rev[i]]).collect();
        let mut im = vec![0f64; N_FFT];
        let mut len = 2;
        while len <= N_FFT {
            let step = N_FFT / len;
            for start in (0..N_FFT).step_by(len) {
                for k in 0..len / 2 {
                    let (c, s) = (self.cos[k * step], self.sin[k * step]);
                    let (a, b) = (start + k, start + k + len / 2);
                    let tr = re[b] * c - im[b] * s;
                    let ti = re[b] * s + im[b] * c;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len *= 2;
        }
        for (k, o) in out.iter_mut().enumerate().take(BINS) {
            let mag = re[k].hypot(im[k]) as f32;
            *o = mag * mag;
        }
    }
}

impl Default for Fft {
    fn default() -> Self {
        Fft::new()
    }
}

/// NeMo features for 16 kHz audio: (128 × frames, row-major by mel) and the valid length.
pub fn features(audio: &[f32], bank: &[f32], fft: &Fft) -> (Vec<f32>, usize, usize) {
    let n = audio.len();
    // Pre-emphasis (float32, as numpy does it), then 256 zeros on each side.
    let mut x = vec![0f32; n + N_FFT];
    for i in 0..n {
        let prev = if i == 0 { 0.0 } else { audio[i - 1] };
        x[N_FFT / 2 + i] = audio[i] - 0.97f32 * prev;
    }
    // np.hanning(400) (symmetric), centred in 512.
    let pad = (N_FFT - WIN) / 2;
    let window: Vec<f64> = (0..N_FFT).map(|i| if i < pad || i >= pad + WIN { 0.0 } else { 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i - pad) as f64 / (WIN - 1) as f64).cos() }).collect();
    let frames = n / HOP + 1;
    let valid = n / HOP;
    let mut logmel = vec![0f32; frames * MELS];
    let mut frame = vec![0f64; N_FFT];
    let mut power = vec![0f32; BINS];
    let guard = 2f32.powi(-24);
    for t in 0..frames {
        for i in 0..N_FFT {
            frame[i] = x[t * HOP + i] as f64 * window[i];
        }
        fft.power(&frame, &mut power);
        for m in 0..MELS {
            let mut s = 0f32;
            for b in 0..BINS {
                s += power[b] * bank[b * MELS + m];
            }
            logmel[t * MELS + m] = (s + guard).ln();
        }
    }
    // Per-utterance normalisation over the valid frames; the rest is zero.
    let mut out = vec![0f32; MELS * frames];
    for m in 0..MELS {
        let mean = ((0..valid).map(|t| logmel[t * MELS + m] as f64).sum::<f64>() / valid as f64) as f32;
        let var = ((0..valid).map(|t| ((logmel[t * MELS + m] - mean) as f64).powi(2)).sum::<f64>() / (valid as f64 - 1.0)) as f32;
        let sd = var.sqrt() + 1e-5;
        for t in 0..valid {
            out[m * frames + t] = (logmel[t * MELS + m] - mean) / sd;
        }
    }
    (out, frames, valid)
}

// --- Decoding ----------------------------------------------------------------------------------------

/// Vocabulary: id → token (▁ as a space), and the blank's id.
pub fn vocab(text: &str) -> Result<(Vec<String>, usize), String> {
    let mut v: Vec<(usize, String)> = vec![];
    for line in text.lines().filter(|l| !l.is_empty()) {
        let (tok, id) = line.rsplit_once(' ').ok_or("vocab.txt: a line without an id")?;
        v.push((id.parse().map_err(|_| "vocab.txt: a bad id")?, tok.replace('\u{2581}', " ")));
    }
    v.sort_by_key(|p| p.0);
    let blank = v.iter().find(|p| p.1 == "<blk>").map(|p| p.0).ok_or("vocab.txt has no <blk>")?;
    Ok((v.into_iter().map(|p| p.1).collect(), blank))
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// One run of the decoder-joint on an encoder frame, with the prediction state it holds.
pub trait Decoder {
    /// Logits for frame `t` after `last`: the vocabulary, then the durations.
    fn step(&mut self, t: usize, last: usize) -> Result<Vec<f32>, String>;
    /// Keeps the state the last step made (a token was emitted).
    fn commit(&mut self);
}

/// Greedy TDT decoding over `len` encoder frames (onnx-asr's transducer loop).
pub fn tdt_decode(len: usize, vocab_size: usize, blank: usize, d: &mut dyn Decoder) -> Result<Vec<usize>, String> {
    let mut tokens: Vec<usize> = vec![];
    let (mut t, mut emitted) = (0, 0);
    while t < len {
        let out = d.step(t, tokens.last().copied().unwrap_or(blank))?;
        let token = argmax(&out[..vocab_size]);
        let skip = argmax(&out[vocab_size..]);
        if token != blank {
            d.commit();
            tokens.push(token);
            emitted += 1;
        }
        if skip > 0 {
            t += skip;
            emitted = 0;
        } else if token == blank || emitted == MAX_TOKENS_PER_STEP {
            t += 1;
            emitted = 0;
        }
    }
    Ok(tokens)
}

static SPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\A\s|\s\B|(\s)\b").unwrap());

/// Tokens → text, spaces tidied as onnx-asr does, then trimmed.
pub fn text(tokens: &[usize], vocab: &[String]) -> String {
    let joined: String = tokens.iter().filter_map(|&t| vocab.get(t)).map(String::as_str).collect();
    SPACE.replace_all(&joined, |c: &regex::Captures| if c.get(1).is_some() { " " } else { "" }).trim().to_string()
}
