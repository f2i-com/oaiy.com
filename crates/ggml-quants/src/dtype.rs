//! GGML element / block dtype enum. Mirrors `enum ggml_type` upstream.
//!
//! Wire values MUST match: this is part of the GGUF on-disk format.

use crate::error::QuantError;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0   = 34,
    TQ2_0   = 35,
    /// 2 bits a weight, a scale a block of 64 (the GSQ-RCO GGUFs' routed experts).
    Q2_0    = 42,
}

#[derive(Debug, thiserror::Error)]
#[error("unknown ggml dtype tag {0}")]
pub struct UnknownDtype(pub u32);

impl GgmlType {
    pub fn from_u32(v: u32) -> std::result::Result<Self, UnknownDtype> {
        Ok(match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1_M,
            30 => Self::BF16,
            31 => Self::Q4_0_4_4,
            32 => Self::Q4_0_4_8,
            33 => Self::Q4_0_8_8,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            42 => Self::Q2_0,
            other => return Err(UnknownDtype(other)),
        })
    }

    /// Number of elements per quantization block. 1 for non-block formats.
    pub const fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => 256,
            IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ1_S | IQ1_M => 256,
            IQ4_NL => 32,
            IQ4_XS => 256,
            Q4_0_4_4 | Q4_0_4_8 | Q4_0_8_8 => 32,
            TQ1_0 | TQ2_0 => 256,
            Q2_0 => 64,
        }
    }

    /// Bytes per quantization block (per element for non-block formats).
    pub const fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 => 2,
            BF16 => 2,
            F64 => 8,
            I8 => 1,
            I16 => 2,
            I32 => 4,
            I64 => 8,
            Q4_0 => 2 + 16,
            Q4_1 => 2 + 2 + 16,
            Q5_0 => 2 + 4 + 16,
            Q5_1 => 2 + 2 + 4 + 16,
            Q8_0 => 2 + 32,
            Q8_1 => 4 + 4 + 32,
            Q2_K => 16 + 64 + 2 + 2,
            Q3_K => 32 + 64 + 12 + 2,
            Q4_K => 2 + 2 + 12 + 128,
            Q5_K => 2 + 2 + 12 + 32 + 128,
            Q6_K => 128 + 64 + 16 + 2,
            Q8_K => 4 + 256 + 16 * 2,
            IQ2_XXS => 2 + 64,
            IQ2_XS  => 2 + 64 + 16,
            IQ2_S   => 2 + 64 + 16 + 8,
            IQ3_XXS => 2 + 64 + 32,
            IQ3_S   => 2 + 64 + 32 + 4 + 8,
            IQ1_S   => 2 + 32 + 8,
            IQ1_M   => 32 + 8 + 16,
            IQ4_NL  => 2 + 16,
            IQ4_XS  => 2 + 2 + 4 + 128,   // d + scales_h + scales_l[QK_K/64] + qs[QK_K/2]
            Q4_0_4_4 | Q4_0_4_8 | Q4_0_8_8 => 18,
            TQ1_0   => 2 + 53,
            TQ2_0   => 2 + 64,
            Q2_0    => 2 + 16,
        }
    }

    pub const fn is_quantized(self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::BF16 | Self::F64
                       | Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "F32", F16 => "F16", BF16 => "BF16", F64 => "F64",
            I8 => "I8", I16 => "I16", I32 => "I32", I64 => "I64",
            Q4_0 => "Q4_0", Q4_1 => "Q4_1",
            Q5_0 => "Q5_0", Q5_1 => "Q5_1",
            Q8_0 => "Q8_0", Q8_1 => "Q8_1",
            Q2_K => "Q2_K", Q3_K => "Q3_K", Q4_K => "Q4_K",
            Q5_K => "Q5_K", Q6_K => "Q6_K", Q8_K => "Q8_K",
            IQ2_XXS => "IQ2_XXS", IQ2_XS => "IQ2_XS", IQ2_S => "IQ2_S",
            IQ3_XXS => "IQ3_XXS", IQ3_S => "IQ3_S",
            IQ1_S => "IQ1_S", IQ1_M => "IQ1_M",
            IQ4_NL => "IQ4_NL", IQ4_XS => "IQ4_XS",
            Q4_0_4_4 => "Q4_0_4_4", Q4_0_4_8 => "Q4_0_4_8", Q4_0_8_8 => "Q4_0_8_8",
            TQ1_0 => "TQ1_0", TQ2_0 => "TQ2_0",
            Q2_0 => "Q2_0",
        }
    }
}

/// Convert a `QuantError` for an unsupported dtype back into a u32 — used by
/// callers that need to surface the wire tag for diagnostics.
pub fn dtype_to_tag(t: GgmlType) -> u32 { t as u32 }

#[allow(dead_code)]
fn _ensure_quanterror_compiles(_: QuantError) {}
