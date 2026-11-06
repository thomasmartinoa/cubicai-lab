//! Every decoder against gguf-py's reference dequantisation (fixtures made by `fixtures/make.py`).

use cubic_gguf::GgmlType;
use cubic_gguf::dequant::{can_dequantize, dequantize};
use std::path::Path;

fn check(name: &str, ty: GgmlType) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let raw = std::fs::read(dir.join(format!("{name}.bin"))).unwrap();
    let want: Vec<f32> = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert!(can_dequantize(ty), "{name} should be decodable");
    assert_eq!(ty.bytes_for(want.len() as u64), Some(raw.len() as u64), "{name} block size");
    let mut got = vec![f32::NAN; want.len()];
    assert!(dequantize(ty, &raw, &mut got), "{name} refused");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(g == w || (g - w).abs() <= 1e-6 * w.abs().max(1.0), "{name}[{i}]: got {g}, gguf-py says {w}");
    }
}

macro_rules! reference_tests {
    ($($test:ident => $ty:ident),* $(,)?) => {
        $( #[test] fn $test() { check(stringify!($ty), GgmlType::$ty); } )*
    };
}

reference_tests! {
    q4_0 => Q4_0, q4_1 => Q4_1, q5_0 => Q5_0, q5_1 => Q5_1, q8_0 => Q8_0,
    q2_k => Q2_K, q3_k => Q3_K, q4_k => Q4_K, q5_k => Q5_K, q6_k => Q6_K,
    iq4_nl => IQ4_NL, iq4_xs => IQ4_XS, tq2_0 => TQ2_0, mxfp4 => MXFP4,
    bf16 => BF16, f16 => F16,
}

#[test]
fn codebook_types_are_declined_not_garbled() {
    for ty in [GgmlType::IQ2_XXS, GgmlType::IQ3_S, GgmlType::IQ1_M] {
        assert!(!can_dequantize(ty));
        let (_, block) = ty.layout().unwrap();
        let mut out = vec![0f32; 256];
        assert!(!dequantize(ty, &vec![0u8; block as usize], &mut out));
    }
}

#[test]
fn block_sizes_match_ggml() {
    // (type, elements per block, bytes per block) from ggml.c's type_traits.
    for (ty, n, b) in [
        (GgmlType::F32, 1, 4), (GgmlType::F16, 1, 2), (GgmlType::BF16, 1, 2), (GgmlType::Q4_0, 32, 18),
        (GgmlType::Q8_0, 32, 34), (GgmlType::Q4_K, 256, 144), (GgmlType::Q6_K, 256, 210),
        (GgmlType::MXFP4, 32, 17), (GgmlType::IQ4_XS, 256, 136), (GgmlType::TQ2_0, 256, 66),
    ] {
        assert_eq!(ty.layout(), Some((n, b)), "{ty:?}");
    }
    assert!((GgmlType::Q4_K.bits_per_weight().unwrap() - 4.5).abs() < 1e-9);
    assert_eq!(GgmlType::from_id(39), GgmlType::MXFP4);
    assert!(matches!(GgmlType::from_id(999), GgmlType::Unknown(999)));
}
