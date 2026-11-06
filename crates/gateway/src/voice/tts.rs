//! Kokoro's parts that aren't ONNX, as kokoro-onnx 0.4.7 does them: phonemes → token ids, batches
//! of at most 510 phonemes, voice styles from the voices file, trimming the silence around each
//! batch (librosa's `trim`), and float → 16-bit PCM.

use std::collections::HashMap;

pub const MAX_PHONEMES: usize = 510;
pub const STYLE: usize = 256;

/// kokoro-onnx's vocabulary (its config.json): phoneme → token id.
pub const VOCAB: [(char, i64); 114] = [('\u{3b}', 1), ('\u{3a}', 2), ('\u{2c}', 3), ('\u{2e}', 4), ('\u{21}', 5), ('\u{3f}', 6), ('\u{2014}', 9), ('\u{2026}', 10), ('\u{22}', 11), ('\u{28}', 12), ('\u{29}', 13), ('\u{201c}', 14), ('\u{201d}', 15), ('\u{20}', 16), ('\u{303}', 17), ('\u{2a3}', 18), ('\u{2a5}', 19), ('\u{2a6}', 20), ('\u{2a8}', 21), ('\u{1d5d}', 22), ('\u{ab67}', 23), ('\u{41}', 24), ('\u{49}', 25), ('\u{4f}', 31), ('\u{51}', 33), ('\u{53}', 35), ('\u{54}', 36), ('\u{57}', 39), ('\u{59}', 41), ('\u{1d4a}', 42), ('\u{61}', 43), ('\u{62}', 44), ('\u{63}', 45), ('\u{64}', 46), ('\u{65}', 47), ('\u{66}', 48), ('\u{68}', 50), ('\u{69}', 51), ('\u{6a}', 52), ('\u{6b}', 53), ('\u{6c}', 54), ('\u{6d}', 55), ('\u{6e}', 56), ('\u{6f}', 57), ('\u{70}', 58), ('\u{71}', 59), ('\u{72}', 60), ('\u{73}', 61), ('\u{74}', 62), ('\u{75}', 63), ('\u{76}', 64), ('\u{77}', 65), ('\u{78}', 66), ('\u{79}', 67), ('\u{7a}', 68), ('\u{251}', 69), ('\u{250}', 70), ('\u{252}', 71), ('\u{e6}', 72), ('\u{3b2}', 75), ('\u{254}', 76), ('\u{255}', 77), ('\u{e7}', 78), ('\u{256}', 80), ('\u{f0}', 81), ('\u{2a4}', 82), ('\u{259}', 83), ('\u{25a}', 85), ('\u{25b}', 86), ('\u{25c}', 87), ('\u{25f}', 90), ('\u{261}', 92), ('\u{265}', 99), ('\u{268}', 101), ('\u{26a}', 102), ('\u{29d}', 103), ('\u{26f}', 110), ('\u{270}', 111), ('\u{14b}', 112), ('\u{273}', 113), ('\u{272}', 114), ('\u{274}', 115), ('\u{f8}', 116), ('\u{278}', 118), ('\u{3b8}', 119), ('\u{153}', 120), ('\u{279}', 123), ('\u{27e}', 125), ('\u{27b}', 126), ('\u{281}', 128), ('\u{27d}', 129), ('\u{282}', 130), ('\u{283}', 131), ('\u{288}', 132), ('\u{2a7}', 133), ('\u{28a}', 135), ('\u{28b}', 136), ('\u{28c}', 138), ('\u{263}', 139), ('\u{264}', 140), ('\u{3c7}', 142), ('\u{28e}', 143), ('\u{292}', 147), ('\u{294}', 148), ('\u{2c8}', 156), ('\u{2cc}', 157), ('\u{2d0}', 158), ('\u{2b0}', 162), ('\u{2b2}', 164), ('\u{2193}', 169), ('\u{2192}', 171), ('\u{2197}', 172), ('\u{2198}', 173), ('\u{1d7b}', 177)];

pub fn vocab_id(c: char) -> Option<i64> {
    VOCAB.iter().find(|(k, _)| *k == c).map(|(_, v)| *v)
}

/// What the Tokenizer keeps of phonemizer's output: only phonemes Kokoro knows, trimmed.
pub fn keep_known(phonemes: &str) -> String {
    phonemes.chars().filter(|c| vocab_id(*c).is_some()).collect::<String>().trim().to_string()
}

pub fn tokenize(phonemes: &str) -> Vec<i64> {
    phonemes.chars().filter_map(vocab_id).collect()
}

/// Batches of phonemes, cut at punctuation (`Kokoro._split_phonemes`; lengths in characters).
pub fn split_phonemes(phonemes: &str) -> Vec<String> {
    // re.split(r"([.,!?;])", …): the pieces with the marks kept as their own parts.
    let mut parts = vec![];
    let mut cur = String::new();
    for c in phonemes.chars() {
        if ".,!?;".contains(c) {
            parts.push(std::mem::take(&mut cur));
            parts.push(c.to_string());
        } else {
            cur.push(c);
        }
    }
    parts.push(cur);
    let mut batches = vec![];
    let mut batch = String::new();
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if batch.chars().count() + part.chars().count() + 1 >= MAX_PHONEMES {
            batches.push(batch.trim().to_string());
            batch = part.to_string();
        } else if ".,!?;".contains(part) {
            batch.push_str(part);
        } else {
            if !batch.is_empty() {
                batch.push(' ');
            }
            batch.push_str(part);
        }
    }
    if !batch.is_empty() {
        batches.push(batch.trim().to_string());
    }
    batches
}

/// librosa.effects.trim(y, top_db=60, frame_length=2048, hop_length=512): what's left between the
/// first and last frames within 60 dB of the loudest.
pub fn trim(y: &[f32]) -> &[f32] {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    let n = y.len();
    // Centred frames over the zero-padded signal.
    let frames = 1 + n / HOP;
    let at = |i: isize| if i < 0 || i as usize >= n { 0.0 } else { y[i as usize] as f64 };
    let rms: Vec<f64> = (0..frames)
        .map(|f| {
            let start = (f * HOP) as isize - (FRAME / 2) as isize;
            let power = (0..FRAME as isize).map(|k| {
                let v = at(start + k) as f32;
                (v * v) as f64
            });
            (power.sum::<f64>() as f32 / FRAME as f32).sqrt() as f64
        })
        .collect();
    let amin = 1e-10f64;
    let peak = rms.iter().cloned().fold(0.0, f64::max);
    let reference = 10.0 * (peak * peak).max(amin).log10();
    let loud: Vec<usize> = (0..frames).filter(|&f| 10.0 * (rms[f] * rms[f]).max(amin).log10() - reference > -60.0).collect();
    match (loud.first(), loud.last()) {
        (Some(&a), Some(&b)) => &y[a * HOP..n.min((b + 1) * HOP)],
        _ => &y[..0],
    }
}

pub fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes()).collect()
}

/// The voice styles: each voice is 510 × 256 float32 (`.npy` files in an uncompressed zip).
pub fn read_voices(data: &[u8]) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut voices = HashMap::new();
    let mut at = 0;
    let u16_at = |p: usize| data.get(p..p + 2).map(|b| u16::from_le_bytes([b[0], b[1]]) as usize).ok_or("voices file is cut short");
    let u32_at = |p: usize| data.get(p..p + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize).ok_or("voices file is cut short");
    while data.get(at..at + 4) == Some(b"PK\x03\x04") {
        let (method, mut size) = (u16_at(at + 8)?, u32_at(at + 18)?);
        let (name_len, extra_len) = (u16_at(at + 26)?, u16_at(at + 28)?);
        let name = String::from_utf8_lossy(&data[at + 30..at + 30 + name_len]).into_owned();
        let extra = &data[at + 30 + name_len..at + 30 + name_len + extra_len];
        if size == 0xFFFF_FFFF {
            // Zip64: the real size is in the extra field.
            let mut e = 0;
            while e + 4 <= extra.len() {
                let (id, len) = (u16::from_le_bytes([extra[e], extra[e + 1]]), u16::from_le_bytes([extra[e + 2], extra[e + 3]]) as usize);
                if id == 1 && len >= 16 {
                    size = u64::from_le_bytes(extra[e + 12..e + 20].try_into().unwrap()) as usize;
                }
                e += 4 + len;
            }
        }
        if method != 0 {
            return Err("voices file is compressed".into());
        }
        let start = at + 30 + name_len + extra_len;
        let body = data.get(start..start + size).ok_or("voices file is cut short")?;
        if let Some(voice) = name.strip_suffix(".npy") {
            voices.insert(voice.to_string(), npy_f32(body)?);
        }
        at = start + size;
    }
    if voices.is_empty() {
        return Err("no voices in the voices file".into());
    }
    Ok(voices)
}

/// A little-endian float32 C-order `.npy` array's values.
pub fn npy_f32(b: &[u8]) -> Result<Vec<f32>, String> {
    if !b.starts_with(b"\x93NUMPY") || b.len() < 10 {
        return Err("not a .npy array".into());
    }
    let (header_len, offset) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize, 12) };
    let header = String::from_utf8_lossy(&b[offset..offset + header_len]);
    if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
        return Err("not a float32 C-order array".into());
    }
    Ok(b[offset + header_len..].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
}
