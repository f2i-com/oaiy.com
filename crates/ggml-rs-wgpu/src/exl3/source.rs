//! A projection's own kernels ([`shader`]): the lanes of a tile's decode (the general form and each rate's own),
//! the one-row kernel and a prompt's many rows'.

/// The parts both kernels share: the parameters, the bindings, and where a thread's weight starts in its tile.
/// A one-row kernel's tile loop as a warp decodes it: lane `l` of a warp holds codes `8l..8l + 8` of a tile (the
/// trellis' order), which are weights `(r, c)` for `r` in `2 (l % 4) + {0, 1, 8, 9}` and `c` in `l / 4 + {0, 8}`; it
/// loads the four words its codes lie in and its four inputs once a tile, and sums its two columns' products (in
/// code order). A workgroup's eight warps take a split's tiles in turn, and column `c`'s sum is its four lanes' in
/// each warp, the warps in order. `x_at` is the WGSL of the input's index for row `r` of tile `kt`, `words_at` of a
/// tile's first word, `out` of the column's sum's place (`total` the sum, `c` the column).
fn one_lanes(x_at: &str, words_at: &str, out: &str) -> String {
    one_lanes_with(x_at, words_at, out, None)
}

/// The rates a tile may be written at, in 16-bit words a tile: 1 to 4 bits a weight by halves, and 5 to 8.
pub(crate) const RATES: [usize; 11] = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128];

/// What differs between [`one_lanes`]' forms ([`one_lanes_with`]): how a thread writes its table entry, what a lane
/// reads of the table, its loads of a tile's words, and its eight codes.
pub(super) struct Lanes {
    pub(super) place: String,
    pub(super) held: String,
    pub(super) loads: String,
    pub(super) codes: Vec<String>,
}

pub(super) fn lanes_general() -> Lanes {
    Lanes {
        place: "    places[t] = place(t, tw, first);\n".into(),
        held: "    var at: array<u32, 8>;\n".to_string() + &(0..8).map(|jj| format!("    at[{jj}] = places[8u * l + {jj}u];\n")).collect::<String>(),
        loads: (0..4).map(|i| format!("        let q{i} = words[tile + {}];\n", ["w0", "o1", "o2", "o3"][i])).collect(),
        codes: (0..8).map(|jj| format!("code_in(q0, q1, q2, q3, at[{jj}])")).collect(),
    }
}

/// Where a lane's code `j` starts after its first, in bits, at `tw` words a tile (as `window` places them).
fn code_offset(tw: usize, j: usize) -> usize {
    j * (tw / 16) + if tw % 16 == 8 { j.div_ceil(2) } else { 0 }
}

/// The furthest into its word any lane's first code starts, in bits, at `tw` words a tile.
fn furthest_start(tw: usize) -> usize {
    let total = tw * 16;
    (0..32).map(|l| ((8 * l + 1) * (tw / 16) + if tw % 16 == 8 { (8 * l + 1) / 2 } else { 0 } + total - 16) % total % 32).max().unwrap()
}

/// A lane's loop for tiles of `tw` words alone, each code's place written into the kernel. A lane's code `j` starts
/// `j b` bits after its first (and `(j + 1) / 2` more at a half rate), whatever the lane; only where the first code
/// starts in its word differs between lanes. So a lane shifts its words to its first code once a tile (`a0`, `a1`,
/// `a2`: the stream's bits 0, 32 and 64 on; `m0`, `m1`: bits 16 and 48 on), and a code is one of those shifted by a
/// constant, where the general loop picks each code's view of three by two selects and shifts it by a number it read.
/// It loads only the words some lane's codes reach (two at 3 and 4 bits a weight, where the general loop loads four).
/// The same codes, so the same sums.
pub(super) fn lanes_fixed(tw: usize) -> Lanes {
    // code j's view (0: a0, 1: m0, 2: a1, 3: m1, 4: a2, 32 bits of the lane's stream from 16 times its number) and
    // how far the code is above the view's low end
    let at: Vec<(usize, usize)> = (0..8)
        .map(|j| {
            let f = code_offset(tw, j);
            let view = if f <= 16 { 0 } else { (f - 1) / 16 };
            (view, 16 * view + 16 - f)
        })
        .collect();
    let last = at.iter().map(|&(view, _)| view).max().unwrap();
    assert!(last <= 4, "a lane's codes lie within 96 bits of its first");
    // the words some lane reaches, and the aligned words the views are made of
    let words = (furthest_start(tw) + code_offset(tw, 7) + 16).div_ceil(32);
    let aligned = last.div_ceil(2) + 1;
    assert!((aligned..=4).contains(&words), "a lane's views lie in the words it loads");
    let mut loads: String = (0..words).map(|i| format!("        let q{i} = words[tile + {}];\n", ["w0", "o1", "o2", "o3"][i])).collect();
    for i in 0..aligned {
        // (a shift of 32 is no shift at all, so the next word's part goes down by one and then by `31 - into`)
        loads += &if i + 1 < words { format!("        let a{i} = (q{i} << into) | ((q{} >> 1u) >> rest);\n", i + 1) } else { format!("        let a{i} = q{i} << into;\n") };
    }
    for i in 0..2 {
        if at.iter().any(|&(view, _)| view == 2 * i + 1) {
            loads += &format!("        let m{i} = (a{i} << 16u) | (a{} >> 16u);\n", i + 1);
        }
    }
    Lanes {
        place: "    places[t] = 48u - window(t, tw).y;\n".into(),
        held: "    let into = places[8u * l];\n    let rest = 31u - into;\n".into(),
        loads,
        codes: at.iter().map(|&(view, shift)| format!("{} >> {shift}u", ["a0", "m0", "a1", "m1", "a2"][view])).collect(),
    }
}

pub(super) fn one_lanes_with(x_at: &str, words_at: &str, out: &str, fixed: Option<usize>) -> String {
    let Lanes { place, held, loads, codes } = fixed.map_or_else(lanes_general, lanes_fixed);
    let codes: String = codes
        .iter()
        .enumerate()
        .map(|(jj, code)| {
            let acc = if jj < 4 { "lo" } else { "hi" };
            format!("            {acc} = {acc} + xv[{}] * decode_at({code});\n", jj % 4)
        })
        .collect();
    format!(
        r#"
var<workgroup> red: array<vec2<f32>, 256>;
// every code's place in a tile (`place`; in a kernel written for one rate, how far into its word it starts, read of
// a lane's first code alone), the workgroup's threads one each: the same in every tile
var<workgroup> places: array<u32, 256>;
var<workgroup> firsts: array<u32, 32>;
// a code's value by its bytes' sum (0 to 1,020), the workgroup's threads four each: a weight is then a read where it
// was a conversion, a product and a rounding to f16
var<workgroup> values: array<f32, 1024>;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

// The 16-bit window of code `i` of a tile of `tw` words: (its first word, its shift) as `weight` finds them.
fn window(i: u32, tw: u32) -> vec2<u32> {{
    let nw = tw / 2u;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec2<u32>(start / 32u, 48u - start % 32u);
}}

// Code `i`'s view and its shift, packed, for a lane whose first word is `w0`: which 32 bits of the lane's four
// words its window lies in (view 2 d: word d; view 2 d + 1: word d's low half and the next's high half), and how
// far the code is above the view's low end. A lane's eight codes start within 96 bits of its first word (8 bits
// a weight at most), so views 0 to 5.
fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    let o = 32u * ((wd.x + nw - w0) % nw) + (48u - wd.y);
    return (o >> 4u) | ((16u - (o & 15u)) << 8u);
}}

// A code from the four words `q0..q3` (from a lane's first) at its packed place: its view, shifted (its low 16
// bits: `decode_at` takes no more). The straddling views are the same for a lane's eight codes.
fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let odd = (at & 1u) == 1u;
    let p0 = select(q0, (q0 << 16u) | (q1 >> 16u), odd);
    let p1 = select(q1, (q1 << 16u) | (q2 >> 16u), odd);
    let p2 = select(q2, (q2 << 16u) | (q3 >> 16u), odd);
    let hi = (at >> 1u) & 3u;
    return select(select(p0, p1, hi == 1u), p2, hi == 2u) >> (at >> 8u);
}}

fn decode_at(code: u32) -> f32 {{
    let hx = (code & 0xffffu) * 0x83dcd12du;
    return values[dot4U8Packed(hx, 0x01010101u)];
}}

fn lanes(t: u32, nt: u32, ntiles: u32, ks: u32, ke: u32, tw: u32, base: u32) -> vec2<f32> {{
    let warp = t / 32u;
    let l = t % 32u;
    let nw = tw / 2u;
    let rb = 2u * (l % 4u);
    let first = window(8u * (t / 8u), tw).x;
{place}    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    for (var i = 0u; i < 4u; i = i + 1u) {{
        let sum = 4u * t + i;
        values[sum] = round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
    }}
    workgroupBarrier();
    let w0 = firsts[l];
    let o1 = (w0 + 1u) % nw;
    let o2 = (w0 + 2u) % nw;
    let o3 = (w0 + 3u) % nw;
{held}    var lo = 0.0;
    var hi = 0.0;
    for (var kt = ks + warp; kt < ke; kt = kt + 8u) {{
        let tile = {words_at};
{loads}        let xv = array<f32, 4>(x[{x0}], x[{x1}], x[{x8}], x[{x9}]);
        {{
{codes}        }}
    }}
    return vec2<f32>(lo, hi);
}}

// Column `c`'s sum of a workgroup's lanes (`red`, after its barrier): each warp's four lanes of it, the warps in order.
fn column(c: u32) -> f32 {{
    let half = c / 8u;
    let cl = c % 8u;
    var total = 0.0;
    for (var w = 0u; w < 8u; w = w + 1u) {{
        for (var q = 0u; q < 4u; q = q + 1u) {{
            let v = red[w * 32u + cl * 4u + q];
            total = total + select(v.x, v.y, half == 1u);
        }}
    }}
    return total;
}}
"#,
        x0 = x_at.replace("{r}", "rb"),
        x1 = x_at.replace("{r}", "rb + 1u"),
        x8 = x_at.replace("{r}", "rb + 8u"),
        x9 = x_at.replace("{r}", "rb + 9u"),
    ) + &format!("// {out}\n")
}

const COMMON: &str = r#"
struct Params {
    n: u32,
    k: u32,
    kt0: u32,
    kts: u32,
    tw: u32,
    rows: u32,
    splits: u32,
    slot0: u32,
};
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// f32 to f16 and back, rounding to nearest even: exact for the decoded weights, which are normal f16 values.
fn round_f16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}

// Where weight (r, c)'s 16-bit code sits in a tile of `tw` words: its two words and the shift (the same in every tile).
fn code_at(r: u32, c: u32, tw: u32) -> vec3<u32> {
    let nw = tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let j = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + j;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {
        end = end + (i + 1u) / 2u;
    }
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec3<u32>(start / 32u, (start / 32u + 1u) % nw, 48u - start % 32u);
}

// The weight whose code is in words `a` and `b` at shift `sh` (`code_at`), decoded as mul1.
fn decode_at(a: u32, b: u32, sh: u32) -> f32 {
    let code = select((a << (32u - sh)) | (b >> sh), a >> (sh - 32u), sh >= 32u);
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}

"#;

/// One input row (a decode step): a workgroup of 256 threads takes one 16-wide tile column and a run of its tile rows
/// (one split), its warps the tiles in turn as [`one_lanes`] has them. Each split writes its partial sums to a slot of
/// its own, which the host adds up.
fn one_source() -> String {
    let body = one_lanes("(p.kt0 + kt) * 16u + {r}", "(kt * ntiles + nt) * nw", "");
    format!(
        r#"
struct Params {{
    n: u32,
    k: u32,
    kt0: u32,
    kts: u32,
    tw: u32,
    rows: u32,
    splits: u32,
    slot0: u32,
}};
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
{body}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let ntiles = p.n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    let s = wg.z;
    let per = (p.kts + p.splits - 1u) / p.splits;
    let ks = s * per;
    let ke = min(p.kts, ks + per);
    red[t] = lanes(t, nt, ntiles, ks, ke, p.tw, 0u);
    workgroupBarrier();
    if (t < 16u) {{
        part[(p.slot0 + s) * p.n + nt * 16u + t] = column(t);
    }}
}}
"#
    )
}

/// Up to [`ROWS`] input rows (a prompt): each tile's 256 weights are decoded once into the workgroup's memory, and
/// thread `(r, c)` sums column `c` for rows `r` and `r + 16`, sixteen products a tile each. Each split writes its
/// partial sums to a slot of its own, which the host adds up.
const MANY: &str = r#"
var<workgroup> tile: array<u32, 64>;
var<workgroup> xs: array<f32, 512>;
var<workgroup> wt: array<f32, 256>;

// Weight (r, c) of the tile in `tile`: its 16-bit code from the bit stream, decoded as mul1.
fn weight(r: u32, c: u32) -> f32 {
    let nw = p.tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let j = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + j;
    var end = (i + 1u) * (p.tw / 16u);
    if (p.tw % 16u == 8u) {
        end = end + (i + 1u) / 2u;
    }
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    let w0 = start / 32u;
    let sh = 48u - start % 32u;
    let a = tile[w0];
    let b = tile[(w0 + 1u) % nw];
    var code: u32;
    if (sh >= 32u) {
        code = a >> (sh - 32u);
    } else {
        code = (a << (32u - sh)) | (b >> sh);
    }
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let ntiles = p.n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {
        return;
    }
    let s = wg.z;
    let r = t / 16u;
    let c = t % 16u;
    let nw = p.tw / 2u;
    let per = (p.kts + p.splits - 1u) / p.splits;
    let ks = s * per;
    let ke = min(p.kts, ks + per);
    let a = r < p.rows;
    let b = r + 16u < p.rows;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var kt = ks; kt < ke; kt = kt + 1u) {
        if (t < nw) {
            tile[t] = words[(kt * ntiles + nt) * nw + t];
        }
        for (var q = t; q < p.rows * 16u; q = q + 256u) {
            xs[q] = x[(q / 16u) * p.k + (p.kt0 + kt) * 16u + (q % 16u)];
        }
        workgroupBarrier();
        wt[t] = weight(r, c);
        workgroupBarrier();
        if (a) {
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                acc0 = acc0 + xs[r * 16u + kk] * wt[kk * 16u + c];
            }
        }
        if (b) {
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                acc1 = acc1 + xs[(r + 16u) * 16u + kk] * wt[kk * 16u + c];
            }
        }
        workgroupBarrier();
    }
    if (a) {
        part[((p.slot0 + s) * p.rows + r) * p.n + nt * 16u + c] = acc0;
    }
    if (b) {
        part[((p.slot0 + s) * p.rows + r + 16u) * p.n + nt * 16u + c] = acc1;
    }
}
"#;

/// The kernel for one row, and the one for several, as WGSL.
pub fn shader(many: bool) -> String {
    if many {
        format!("{COMMON}{MANY}")
    } else {
        one_source()
    }
}
