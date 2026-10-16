//! Each type's `dequant(bb, sub)` in WGSL (the 32 values of sub-block `sub` of the block at byte `bb`, into `v`),
//! ggml's own decodes line by line: the plain types and the K-quants, then the grid types with their tables.

use super::*;

// Each `dequant(bb, sub)`: `bb` is the block's first byte, `sub` which 32 of its
// values to produce.

pub(super) const Q4_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    // (on the GPU its nibbles after a 2-byte gap: `padded_block`)
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 4u + i);
        v[i] = (f32(q & 15u) - 8.0) * d;
        v[i + 16u] = (f32(q >> 4u) - 8.0) * d;
    }
}
"#;

// Q2_0: a scale, then 64 2-bit codes four a byte (a code of 0..3 the scale's -1, 0, +1 and +2).
pub(super) const Q2_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    for (var i = 0u; i < 8u; i++) {
        let q = byte(bb + 2u + sub * 8u + i);
        v[i * 4u] = (f32(q & 3u) - 1.0) * d;
        v[i * 4u + 1u] = (f32((q >> 2u) & 3u) - 1.0) * d;
        v[i * 4u + 2u] = (f32((q >> 4u) & 3u) - 1.0) * d;
        v[i * 4u + 3u] = (f32(q >> 6u) - 1.0) * d;
    }
}
"#;

pub(super) const Q4_1: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let mn = f16at(bb + 2u);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 4u + i);
        v[i] = f32(q & 15u) * d + mn;
        v[i + 16u] = f32(q >> 4u) * d + mn;
    }
}
"#;

pub(super) const Q5_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let qh = u32at(bb + 2u);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 6u + i);
        let lo = (q & 15u) | (((qh >> i) & 1u) << 4u);
        let hi = (q >> 4u) | (((qh >> (i + 16u)) & 1u) << 4u);
        v[i] = (f32(lo) - 16.0) * d;
        v[i + 16u] = (f32(hi) - 16.0) * d;
    }
}
"#;

pub(super) const Q5_1: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let mn = f16at(bb + 2u);
    let qh = u32at(bb + 4u);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 8u + i);
        let lo = (q & 15u) | (((qh >> i) & 1u) << 4u);
        let hi = (q >> 4u) | (((qh >> (i + 16u)) & 1u) << 4u);
        v[i] = f32(lo) * d + mn;
        v[i + 16u] = f32(hi) * d + mn;
    }
}
"#;

pub(super) const Q8_0: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    for (var i = 0u; i < 32u; i++) { v[i] = i8of(byte(bb + 2u + i)) * d; }
}
"#;

pub(super) const IQ4_NL: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    for (var i = 0u; i < 16u; i++) {
        let q = byte(bb + 2u + i);
        v[i] = d * KVALUES[q & 15u];
        v[i + 16u] = d * KVALUES[q >> 4u];
    }
}
"#;

// Q4_K / Q5_K 6-bit scale and min for sub-block j (upstream get_scale_min_k4).
pub(super) const SCALE_MIN_K4: &str = r#"
fn scale_min(s: u32, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(byte(s + j) & 63u), f32(byte(s + j + 4u) & 63u));
    }
    let sc = (byte(s + j + 4u) & 15u) | ((byte(s + j - 4u) >> 6u) << 4u);
    let mn = (byte(s + j + 4u) >> 4u) | ((byte(s + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(mn));
}
"#;

pub(super) const Q4_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let dmin = f16at(bb + 2u);
    let sm = scale_min(bb + 4u, sub);
    let d1 = d * sm.x;
    let m1 = dmin * sm.y;
    let q_off = bb + 16u + (sub / 2u) * 32u;
    let shift = (sub & 1u) * 4u;
    for (var l = 0u; l < 32u; l++) {
        v[l] = d1 * f32((byte(q_off + l) >> shift) & 15u) - m1;
    }
}
"#;

pub(super) const Q5_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let dmin = f16at(bb + 2u);
    let sm = scale_min(bb + 4u, sub);
    let d1 = d * sm.x;
    let m1 = dmin * sm.y;
    let q_off = bb + 48u + (sub / 2u) * 32u;
    let shift = (sub & 1u) * 4u;
    let mask = 1u << sub;
    for (var l = 0u; l < 32u; l++) {
        let high = select(0u, 16u, (byte(bb + 16u + l) & mask) != 0u);
        v[l] = d1 * f32(((byte(q_off + l) >> shift) & 15u) + high) - m1;
    }
}
"#;

// Q6_K: sub-block `sub` is quarter `sub % 4` of half `sub / 4`.
pub(super) const Q6_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let h = sub / 4u;
    let qd = sub % 4u;
    let d = f16at(bb + 208u);
    let ql = bb + h * 64u + (qd & 1u) * 32u;
    let qh = bb + 128u + h * 32u;
    let lshift = select(0u, 4u, qd >= 2u);
    let hshift = 2u * qd;
    let sc = bb + 192u + h * 8u + 2u * qd;
    for (var l = 0u; l < 32u; l++) {
        let q = ((byte(ql + l) >> lshift) & 15u) | (((byte(qh + l) >> hshift) & 3u) << 4u);
        v[l] = d * i8of(byte(sc + l / 16u)) * (f32(q) - 32.0);
    }
}
"#;

// Q2_K: sub = 4 * outer + j; halves of 16 values carry their own scale byte.
pub(super) const Q2_K: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb + 80u);
    let dmin = f16at(bb + 82u);
    let q_base = bb + 16u + (sub / 4u) * 32u;
    let shift = 2u * (sub % 4u);
    for (var half = 0u; half < 2u; half++) {
        let sc = byte(bb + 2u * sub + half);
        let dl = d * f32(sc & 15u);
        let ml = dmin * f32(sc >> 4u);
        for (var l = 0u; l < 16u; l++) {
            let q = (byte(q_base + half * 16u + l) >> shift) & 3u;
            v[half * 16u + l] = dl * f32(q) - ml;
        }
    }
}
"#;

// Q3_K: the 12 packed scale bytes unpack to 16 six-bit scales (upstream's aux
// shuffle); the high bit of each 3-bit value is in hmask.
pub(super) const Q3_K: &str = r#"
fn q3_scale(s: u32, k: u32) -> f32 {
    let a0 = u32at(s);
    let a1 = u32at(s + 4u);
    let tmp = u32at(s + 8u);
    var word: u32;
    switch (k / 4u) {
        case 0u: { word = (a0 & 0x0f0f0f0fu) | (((tmp >> 0u) & 0x03030303u) << 4u); }
        case 1u: { word = (a1 & 0x0f0f0f0fu) | (((tmp >> 2u) & 0x03030303u) << 4u); }
        case 2u: { word = ((a0 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 4u) & 0x03030303u) << 4u); }
        default: { word = ((a1 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 6u) & 0x03030303u) << 4u); }
    }
    return i8of((word >> ((k % 4u) * 8u)) & 0xffu);
}
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb + 108u);
    let q_base = bb + 32u + (sub / 4u) * 32u;
    let shift = 2u * (sub % 4u);
    let m = 1u << sub;
    for (var half = 0u; half < 2u; half++) {
        let dl = d * (q3_scale(bb + 96u, 2u * sub + half) - 32.0);
        for (var l = 0u; l < 16u; l++) {
            let i = half * 16u + l;
            let raw = i32((byte(q_base + i) >> shift) & 3u);
            let q = raw - select(4, 0, (byte(bb + i) & m) != 0u);
            v[i] = dl * f32(q);
        }
    }
}
"#;

pub(super) const IQ4_XS: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let scales_h = u16at(bb + 2u);
    let lo4 = (byte(bb + 4u + sub / 2u) >> (4u * (sub & 1u))) & 15u;
    let hi2 = ((scales_h >> (2u * sub)) & 3u) << 4u;
    let dl = d * (f32(lo4 | hi2) - 32.0);
    let q_off = bb + 8u + sub * 16u;
    for (var j = 0u; j < 16u; j++) {
        let q = byte(q_off + j);
        v[j] = dl * KVALUES[q & 15u];
        v[j + 16u] = dl * KVALUES[q >> 4u];
    }
}
"#;

/// A grid type's `dequant` (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M: [`ggml_quants::iq`]'s decodes, a 32 of
/// a block at a time) with the tables it reads as constants beside it: ggml's grids ([`ggml_quants::iq_tables`]; a
/// u64 entry its low word then its high) and the sign patterns. Made once a type.
pub(super) fn iq_grid_dequant(dtype: GgmlType) -> Option<&'static str> {
    use ggml_quants::iq_tables as t;
    use std::sync::OnceLock;
    static SOURCES: [OnceLock<String>; 7] = [const { OnceLock::new() }; 7];
    let words64 = |g: &[u64]| g.iter().flat_map(|v| [*v as u32, (*v >> 32) as u32]).collect::<Vec<u32>>();
    let table = |name: &str, values: &[u32]| format!("const {name} = array<u32, {}>({});\n", values.len(), values.iter().map(|v| format!("{v}u")).collect::<Vec<_>>().join(","));
    let signs = || table("KSIGNS", &t::KSIGNS_IQ2XS.iter().map(|v| *v as u32).collect::<Vec<_>>());
    let (slot, tables, body): (usize, Box<dyn Fn() -> String>, &str) = match dtype {
        GgmlType::IQ2_XXS => (0, Box::new(move || signs() + &table("GRID", &words64(&t::IQ2XXS_GRID))), IQ2_XXS),
        GgmlType::IQ2_XS => (1, Box::new(move || signs() + &table("GRID", &words64(&t::IQ2XS_GRID))), IQ2_XS),
        GgmlType::IQ2_S => (2, Box::new(move || table("GRID", &words64(&t::IQ2S_GRID))), IQ2_S),
        GgmlType::IQ3_XXS => (3, Box::new(move || signs() + &table("GRID", &t::IQ3XXS_GRID)), IQ3_XXS),
        GgmlType::IQ3_S => (4, Box::new(move || table("GRID", &t::IQ3S_GRID)), IQ3_S),
        GgmlType::IQ1_S => (5, Box::new(move || table("GRID", &words64(&t::IQ1S_GRID))), IQ1_S),
        GgmlType::IQ1_M => (6, Box::new(move || table("GRID", &words64(&t::IQ1S_GRID))), IQ1_M),
        _ => return None,
    };
    Some(SOURCES[slot].get_or_init(|| format!("{}{IQ_GROUPS}{body}", tables())).as_str())
}

/// A group of 8 weights into `v[at..]`: magnitudes a byte each from two words (a u64 grid entry's halves, or two u32
/// entries), times `s` and their signs (a bit each of `signs`); and IQ1's, signed bytes plus an offset.
const IQ_GROUPS: &str = r#"
fn put8(at: u32, lo: u32, hi: u32, signs: u32, s: f32) {
    for (var j = 0u; j < 4u; j++) {
        v[at + j] = s * f32((lo >> (8u * j)) & 255u) * select(1.0, -1.0, ((signs >> j) & 1u) == 1u);
        v[at + 4u + j] = s * f32((hi >> (8u * j)) & 255u) * select(1.0, -1.0, ((signs >> (j + 4u)) & 1u) == 1u);
    }
}
fn put1(at: u32, lo: u32, hi: u32, delta: f32, s: f32) {
    for (var j = 0u; j < 4u; j++) {
        v[at + j] = s * (i8of((lo >> (8u * j)) & 255u) + delta);
        v[at + 4u + j] = s * (i8of((hi >> (8u * j)) & 255u) + delta);
    }
}
"#;

const IQ2_XXS: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let at = bb + 2u + 8u * sub;
    let aux = u32at(at + 4u);
    let db = f16at(bb) * (0.5 + f32(aux >> 28u)) * 0.25;
    for (var l = 0u; l < 4u; l++) {
        let i = byte(at + l);
        put8(8u * l, GRID[2u * i], GRID[2u * i + 1u], KSIGNS[(aux >> (7u * l)) & 127u], db);
    }
}
"#;

const IQ2_XS: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let sc = byte(bb + 66u + sub);
    let db0 = d * (0.5 + f32(sc & 15u)) * 0.25;
    let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
    for (var l = 0u; l < 4u; l++) {
        let q = u16at(bb + 2u + 2u * (4u * sub + l));
        let i = q & 511u;
        put8(8u * l, GRID[2u * i], GRID[2u * i + 1u], KSIGNS[q >> 9u], select(db0, db1, l >= 2u));
    }
}
"#;

const IQ2_S: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let d = f16at(bb);
    let qh = byte(bb + 66u + sub);
    let sc = byte(bb + 74u + sub);
    let db0 = d * (0.5 + f32(sc & 15u)) * 0.25;
    let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
    for (var l = 0u; l < 4u; l++) {
        let i = byte(bb + 2u + 4u * sub + l) | ((qh << (8u - 2u * l)) & 0x300u);
        put8(8u * l, GRID[2u * i], GRID[2u * i + 1u], byte(bb + 34u + 4u * sub + l), select(db0, db1, l >= 2u));
    }
}
"#;

const IQ3_XXS: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let aux = u32at(bb + 66u + 4u * sub);
    let db = f16at(bb) * (0.5 + f32(aux >> 28u)) * 0.5;
    for (var l = 0u; l < 4u; l++) {
        let q = bb + 2u + 8u * sub + 2u * l;
        put8(8u * l, GRID[byte(q)], GRID[byte(q + 1u)], KSIGNS[(aux >> (7u * l)) & 127u], db);
    }
}
"#;

const IQ3_S: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let sc = byte(bb + 106u + sub / 2u);
    let db = f16at(bb) * f32(1u + 2u * ((sc >> (4u * (sub % 2u))) & 15u));
    let qh = byte(bb + 66u + sub);
    for (var l = 0u; l < 4u; l++) {
        let q = bb + 2u + 8u * sub + 2u * l;
        put8(8u * l, GRID[byte(q) | ((qh << (8u - 2u * l)) & 256u)], GRID[byte(q + 1u) | ((qh << (7u - 2u * l)) & 256u)], byte(bb + 74u + 4u * sub + l), db);
    }
}
"#;

const IQ1_S: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let qh = u16at(bb + 34u + 2u * sub);
    let dl = f16at(bb) * f32(2u * ((qh >> 12u) & 7u) + 1u);
    let delta = select(0.125, -0.125, (qh & 0x8000u) != 0u);
    for (var l = 0u; l < 4u; l++) {
        let i = byte(bb + 2u + 4u * sub + l) | (((qh >> (3u * l)) & 7u) << 8u);
        put1(8u * l, GRID[2u * i], GRID[2u * i + 1u], delta, dl);
    }
}
"#;

const IQ1_M: &str = r#"
fn dequant(bb: u32, sub: u32) {
    let s0 = u16at(bb + 48u);
    let s1 = u16at(bb + 50u);
    let s2 = u16at(bb + 52u);
    let s3 = u16at(bb + 54u);
    let d = unpack2x16float((s0 >> 12u) | ((s1 >> 8u) & 0x00f0u) | ((s2 >> 4u) & 0x0f00u) | (s3 & 0xf000u)).x;
    let pair = sub / 2u;
    let sc = select(select(s0, s1, pair == 1u), select(s2, s3, pair == 3u), pair >= 2u) >> (6u * (sub % 2u));
    let dl0 = d * f32(2u * (sc & 7u) + 1u);
    let dl1 = d * f32(2u * ((sc >> 3u) & 7u) + 1u);
    for (var l = 0u; l < 4u; l++) {
        let h = byte(bb + 32u + 2u * sub + l / 2u) >> (4u * (l % 2u));
        let i = byte(bb + 4u * sub + l) | ((h & 7u) << 8u);
        put1(8u * l, GRID[2u * i], GRID[2u * i + 1u], select(0.125, -0.125, (h & 8u) != 0u), select(dl0, dl1, l >= 2u));
    }
}
"#;
