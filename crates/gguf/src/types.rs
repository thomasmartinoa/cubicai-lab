//! The ggml tensor types a GGUF file can hold, with their block layout.

use std::fmt;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    /// A type newer than this table; its size comes from the tensor offsets instead.
    Unknown(u32),
}

use GgmlType::*;

impl GgmlType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            39 => MXFP4,
            other => Unknown(other),
        }
    }

    /// The id stored in GGUF files (inverse of `from_id`).
    pub fn id(self) -> u32 {
        if let Unknown(id) = self {
            return id;
        }
        (0..64).find(|&i| GgmlType::from_id(i) == self).unwrap_or(u32::MAX)
    }

    /// Elements per block and bytes per block.
    pub fn layout(self) -> Option<(u64, u64)> {
        Some(match self {
            F32 => (1, 4),
            F16 | BF16 => (1, 2),
            F64 | I64 => (1, 8),
            I32 => (1, 4),
            I16 => (1, 2),
            I8 => (1, 1),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            IQ2_XXS => (256, 66),
            IQ2_XS => (256, 74),
            IQ3_XXS => (256, 98),
            IQ1_S => (256, 50),
            IQ4_NL => (32, 18),
            IQ3_S => (256, 110),
            IQ2_S => (256, 82),
            IQ4_XS => (256, 136),
            IQ1_M => (256, 56),
            TQ1_0 => (256, 54),
            TQ2_0 => (256, 66),
            MXFP4 => (32, 17),
            Unknown(_) => return None,
        })
    }

    pub fn bits_per_weight(self) -> Option<f64> {
        self.layout().map(|(n, b)| b as f64 * 8.0 / n as f64)
    }

    /// Bytes for `n` elements (n a multiple of the block size).
    pub fn bytes_for(self, n: u64) -> Option<u64> {
        self.layout().map(|(bs, ts)| n.div_ceil(bs) * ts)
    }

    pub fn is_float(self) -> bool {
        matches!(self, F32 | F16 | BF16 | F64)
    }

    pub fn name(self) -> String {
        match self {
            Unknown(id) => format!("type {id}"),
            other => format!("{other:?}"),
        }
    }

    /// One line on what the type is, for tooltips.
    pub fn explain(self) -> &'static str {
        match self {
            F32 => "Full 32-bit floats: exact, 4 bytes per weight",
            F16 => "16-bit floats: 2 bytes per weight, practically lossless",
            BF16 => "bfloat16: 16-bit with float32's range, the usual training format",
            Q8_0 => "8-bit blocks of 32 with one scale: near-lossless",
            Q4_0 | Q4_1 => "Old-style 4-bit blocks of 32",
            Q5_0 | Q5_1 => "Old-style 5-bit blocks of 32",
            Q2_K => "2-bit k-quant: super-blocks of 256 with 4-bit scales; lossy",
            Q3_K => "3-bit k-quant: super-blocks of 256 with 6-bit scales",
            Q4_K => "4-bit k-quant: super-blocks of 256, the usual Q4_K_M workhorse",
            Q5_K => "5-bit k-quant: super-blocks of 256",
            Q6_K => "6-bit k-quant: close to Q8 quality",
            IQ4_NL | IQ4_XS => "4-bit with a non-linear value table (tuned to how weights spread)",
            IQ2_XXS | IQ2_XS | IQ2_S => "~2-bit i-quant using lattice codebooks; needs an importance matrix",
            IQ3_XXS | IQ3_S => "~3-bit i-quant using lattice codebooks",
            IQ1_S | IQ1_M => "~1.5-bit i-quant: extreme compression",
            TQ1_0 | TQ2_0 => "Ternary (-1, 0, +1) weights, for BitNet-style models",
            MXFP4 => "Microscale FP4: 4-bit floats sharing a power-of-two scale per 32",
            I8 | I16 | I32 | I64 => "Integers (not weights)",
            _ => "",
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}
