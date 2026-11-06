//! Turns quantized blocks back into floats, following ggml's reference decoders (ggml-quants.c).
//! The codebook i-quants (IQ1/IQ2/IQ3) need large lattice tables and aren't decoded here; their
//! size and placement still show, just not their values.

use crate::types::GgmlType;
use half::{bf16, f16};

const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

pub fn can_dequantize(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(
        ty,
        F32 | F16 | BF16 | F64 | I8 | I16 | I32 | Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q2_K | Q3_K | Q4_K | Q5_K | Q6_K
            | IQ4_NL | IQ4_XS | MXFP4 | TQ2_0
    )
}

#[inline]
fn h(b: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
}

/// Decodes `out.len()` elements (a whole number of blocks) from `data`.
/// Returns false if the type can't be decoded or `data` is too short.
pub fn dequantize(ty: GgmlType, data: &[u8], out: &mut [f32]) -> bool {
    use GgmlType::*;
    let n = out.len();
    let Some((bs, ts)) = ty.layout() else { return false };
    let (bs, ts) = (bs as usize, ts as usize);
    if !n.is_multiple_of(bs) || data.len() < n / bs * ts || !can_dequantize(ty) {
        return false;
    }
    let blocks = data.chunks_exact(ts).zip(out.chunks_exact_mut(bs));
    match ty {
        F32 => blocks.for_each(|(b, y)| y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        F16 => blocks.for_each(|(b, y)| y[0] = h(b, 0)),
        BF16 => blocks.for_each(|(b, y)| y[0] = bf16::from_le_bytes([b[0], b[1]]).to_f32()),
        F64 => blocks.for_each(|(b, y)| y[0] = f64::from_le_bytes(b[..8].try_into().unwrap()) as f32),
        I8 => blocks.for_each(|(b, y)| y[0] = b[0] as i8 as f32),
        I16 => blocks.for_each(|(b, y)| y[0] = i16::from_le_bytes([b[0], b[1]]) as f32),
        I32 => blocks.for_each(|(b, y)| y[0] = i32::from_le_bytes(b[..4].try_into().unwrap()) as f32),
        Q4_0 => blocks.for_each(|(b, y)| {
            let d = h(b, 0);
            let qs = &b[2..18];
            for j in 0..16 {
                y[j] = ((qs[j] & 0xF) as i32 - 8) as f32 * d;
                y[j + 16] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
            }
        }),
        Q4_1 => blocks.for_each(|(b, y)| {
            let (d, m) = (h(b, 0), h(b, 2));
            let qs = &b[4..20];
            for j in 0..16 {
                y[j] = (qs[j] & 0xF) as f32 * d + m;
                y[j + 16] = (qs[j] >> 4) as f32 * d + m;
            }
        }),
        Q5_0 => blocks.for_each(|(b, y)| {
            let d = h(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let qs = &b[6..22];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                y[j] = (((qs[j] & 0xF) as u32 | xh0) as i32 - 16) as f32 * d;
                y[j + 16] = (((qs[j] >> 4) as u32 | xh1) as i32 - 16) as f32 * d;
            }
        }),
        Q5_1 => blocks.for_each(|(b, y)| {
            let (d, m) = (h(b, 0), h(b, 2));
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let qs = &b[8..24];
            for j in 0..16 {
                let xh0 = ((qh >> j) << 4) & 0x10;
                let xh1 = (qh >> (j + 12)) & 0x10;
                y[j] = ((qs[j] & 0xF) as u32 | xh0) as f32 * d + m;
                y[j + 16] = ((qs[j] >> 4) as u32 | xh1) as f32 * d + m;
            }
        }),
        Q8_0 => blocks.for_each(|(b, y)| {
            let d = h(b, 0);
            for j in 0..32 {
                y[j] = b[2 + j] as i8 as f32 * d;
            }
        }),
        Q2_K => blocks.for_each(|(b, y)| {
            let scales = &b[0..16];
            let q = &b[16..80];
            let (d, min) = (h(b, 80), h(b, 82));
            let mut o = 0;
            let mut is = 0;
            for n in 0..2 {
                let q = &q[n * 32..];
                for shift in [0, 2, 4, 6] {
                    for half_ in 0..2 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = min * (sc >> 4) as f32;
                        for l in 0..16 {
                            y[o] = dl * ((q[l + half_ * 16] >> shift) & 3) as f32 - ml;
                            o += 1;
                        }
                    }
                }
            }
        }),
        Q3_K => blocks.for_each(|(b, y)| {
            let hm = &b[0..32];
            let qs = &b[32..96];
            let sc_raw = &b[96..108];
            let d_all = h(b, 108);
            let scales = q3k_scales(sc_raw);
            let mut o = 0;
            let mut is = 0;
            let mut m: u8 = 1;
            for n in 0..2 {
                let q = &qs[n * 32..];
                for shift in [0, 2, 4, 6] {
                    for half_ in 0..2 {
                        let dl = d_all * (scales[is] as i32 - 32) as f32;
                        is += 1;
                        for l in 0..16 {
                            let i = l + half_ * 16;
                            let v = ((q[i] >> shift) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                            y[o] = dl * v as f32;
                            o += 1;
                        }
                    }
                    m <<= 1;
                }
            }
        }),
        Q4_K => blocks.for_each(|(b, y)| {
            let (d, min) = (h(b, 0), h(b, 2));
            let sc = &b[4..16];
            let qs = &b[16..144];
            for j in 0..4 {
                let (s1, m1) = scale_min_k4(2 * j, sc);
                let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                let (d1, m1) = (d * s1 as f32, min * m1 as f32);
                let (d2, m2) = (d * s2 as f32, min * m2 as f32);
                let q = &qs[j * 32..j * 32 + 32];
                let y = &mut y[j * 64..j * 64 + 64];
                for l in 0..32 {
                    y[l] = d1 * (q[l] & 0xF) as f32 - m1;
                    y[l + 32] = d2 * (q[l] >> 4) as f32 - m2;
                }
            }
        }),
        Q5_K => blocks.for_each(|(b, y)| {
            let (d, min) = (h(b, 0), h(b, 2));
            let sc = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            let (mut u1, mut u2) = (1u8, 2u8);
            for j in 0..4 {
                let (s1, m1) = scale_min_k4(2 * j, sc);
                let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                let (d1, m1) = (d * s1 as f32, min * m1 as f32);
                let (d2, m2) = (d * s2 as f32, min * m2 as f32);
                let ql = &qs[j * 32..j * 32 + 32];
                let y = &mut y[j * 64..j * 64 + 64];
                for l in 0..32 {
                    let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    y[l] = d1 * ((ql[l] & 0xF) + hi1) as f32 - m1;
                    y[l + 32] = d2 * ((ql[l] >> 4) + hi2) as f32 - m2;
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }),
        Q6_K => blocks.for_each(|(b, y)| {
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc = &b[192..208];
            let d = h(b, 208);
            for n in 0..2 {
                let ql = &ql[n * 64..];
                let qh = &qh[n * 32..];
                let sc = &sc[n * 8..];
                let y = &mut y[n * 128..n * 128 + 128];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }),
        IQ4_NL => blocks.for_each(|(b, y)| {
            let d = h(b, 0);
            let qs = &b[2..18];
            for j in 0..16 {
                y[j] = d * KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                y[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
            }
        }),
        IQ4_XS => blocks.for_each(|(b, y)| {
            let d = h(b, 0);
            let scales_h = u16::from_le_bytes([b[2], b[3]]);
            let scales_l = &b[4..8];
            let qs = &b[8..136];
            for ib in 0..8 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                let q = &qs[ib * 16..ib * 16 + 16];
                let y = &mut y[ib * 32..ib * 32 + 32];
                for j in 0..16 {
                    y[j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
                    y[j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
                }
            }
        }),
        MXFP4 => blocks.for_each(|(b, y)| {
            // E8M0 scale; the value table is doubled, so it takes half the scale: 2^(e - 128).
            let d = e8m0_half(b[0]);
            let qs = &b[1..17];
            for j in 0..16 {
                y[j] = d * KVALUES_MXFP4[(qs[j] & 0xF) as usize] as f32;
                y[j + 16] = d * KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32;
            }
        }),
        TQ2_0 => blocks.for_each(|(b, y)| {
            let qs = &b[0..64];
            let d = h(b, 64);
            let mut o = 0;
            for j in [0, 32] {
                for l in 0..4 {
                    for m in 0..32 {
                        let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                        y[o] = (q - 1) as f32 * d;
                        o += 1;
                    }
                }
            }
        }),
        _ => return false,
    }
    true
}

fn e8m0_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 { 0x0020_0000 << x } else { ((x as u32) - 1) << 23 };
    f32::from_bits(bits)
}

/// The 6-bit scale and min of sub-block `j` packed in a Q4_K/Q5_K block's 12 scale bytes.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Unpacks Q3_K's 16 six-bit scales from 12 bytes.
fn q3k_scales(raw: &[u8]) -> [u8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0u32; 4];
    for i in 0..3 {
        aux[i] = u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut out = [0u8; 16];
    for (i, a) in aux.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&a.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_block() {
        let mut b = vec![0u8; 34];
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        for j in 0..32 {
            b[2 + j] = (j as i8 - 16) as u8;
        }
        let mut y = [0f32; 32];
        assert!(dequantize(GgmlType::Q8_0, &b, &mut y));
        assert_eq!(y[0], -8.0);
        assert_eq!(y[31], 7.5);
    }

    #[test]
    fn e8m0() {
        assert_eq!(e8m0_half(128), 1.0);
        assert_eq!(e8m0_half(127), 0.5);
        assert_eq!(e8m0_half(130), 4.0);
    }
}
