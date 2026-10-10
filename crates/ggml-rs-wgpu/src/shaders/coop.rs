//! The tensor cores' kernels (WGSL's cooperative matrices): a prompt's matmul by tiles from one template, each
//! type's decode a part put into it; convolutions and NVFP4 by the same.

use super::*;

/// Weight rows (and tokens) a workgroup of [`coop_tiled`] takes.
pub const COOP_TILE: u32 = 128;

/// The steps (of 32 of `k`) a tensor-core matmul sums in f16 before it adds them into its f32 sums: f16 sums run the
/// multiply-adds twice as fast, and over 512 they are within some 0.1% (an int8 activation's own error is near 1%).
const COOP_FOLD: u32 = 32;

/// How many workgroups of 1024 threads run at once (a GPU's SMs or compute units, where one holds one of them; their
/// tensor cores are what a matmul's workgroups share): each works a while (`p[0].x` dependent multiply-adds), and
/// those that start before any has finished are counted (`c[0]`; `c[1]` those finished).
pub const COOP_UNITS_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> unused0: array<u32>;
@group(0) @binding(1) var<storage, read> unused1: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> pad: array<f32, 1024>;

@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    var before = 1u;
    if (li == 0u) { before = atomicLoad(&c[1]); }
    var acc = f32(li) * 1e-3;
    for (var i = 0u; i < p[0].x; i++) { acc = fma(acc, 0.9999, 1e-4); }
    pad[li] = acc;
    workgroupBarrier();
    if (li == 0u) {
        if (before == 0u) { atomicAdd(&c[0], 1u); }
        atomicAdd(&c[1], 1u);
        if (pad[(wg.x * 7u) % 1024u] == 12345.0) { atomicAdd(&c[2], 1u); }
    }
}
"#;

/// The splits of a tensor-core matmul's `steps` (of 32) for `groups` workgroups on a GPU of `units` (its SMs: a
/// workgroup alone on one runs about twice as fast as two together): the fewest of those that take least time, each
/// wave of `units` workgroups taking as long however full, and each split's sums written and read again besides (as
/// much as `420 s / k` of the matmul's own time: the parts mostly in its L2, as [`crate::chain`]'s
/// `measure_coop_splits` finds of Qwen3.8 27B's shapes), at most 8 splits of 8 steps or more; and how many that is
/// with none empty.
pub fn coop_splits(groups: u32, units: u32, steps: u32) -> u32 {
    let slots = units.max(1);
    let k = (steps * 32) as f64;
    let cost = |s: u32| {
        let w = groups * s;
        let waves = w.div_ceil(slots) * slots;
        waves as f64 / w as f64 + if s > 1 { 420.0 * s as f64 / k } else { 0.0 }
    };
    let most = (steps / 8).clamp(1, 8);
    let s = (1..=most).fold(1, |best, s| if cost(s) < cost(best) - 1e-9 { s } else { best });
    steps.div_ceil(steps.div_ceil(s))
}

/// [`coop_tiled`]'s kernel before its type's decode is put in.
const COOP_KERNEL: &str = r#"enable f16;
enable wgpu_cooperative_matrix;
struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, splits: u32, _pad1: u32, }
WEIGHTS_BINDING
@group(0) @binding(1) var<storage, read> x16: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// two steps' weights [row][k] and tokens [token][k] as f16, four to a vec4, a row 10 vec4s apart (8 halves past the
// step's 32: a fragment's 8 rows a load in banks of their own): one step's loaded and stored while the other's are
// multiplied; a subgroup's fragment staged at a tile's edge
const S4: u32 = 10u;
const BUF4: u32 = 1280u;
var<workgroup> wt: array<vec4<f16>, 2560>;
var<workgroup> xt: array<vec4<f16>, 2560>;
var<workgroup> edge: array<f32, 2048>;

DECODE_HELPERS

// a byte as f32, less `o`: its bits in 2^23's mantissa (exact, no conversion)
fn byte_less(w: u32, o: f32) -> vec4<f32> {
    let b = vec4<u32>(w & 255u, (w >> 8u) & 255u, (w >> 16u) & 255u, w >> 24u);
    return bitcast<vec4<f32>>(b | vec4<u32>(0x4b000000u)) - vec4<f32>(8388608.0 + o);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 128u;
    let t0 = wg.y * 128u;
    let sg = li / 32u;
    let lane = li % 32u;
    // the subgroup's 32 rows by 64 tokens of the tile
    let sr = (sg % 4u) * 32u;
    let st = (sg / 4u) * 64u;
    // what this thread decodes and copies: weight row lr's half lh of a step's 32, and token row lr's
    let lr = li / 2u;
    let lh = li % 2u;
    let rr = r0 + lr;
    let rl = min(rr, p.rows - 1u);
    let kx = p.k;
    // the workgroup's split of the steps (`wg.z` of `p.splits`, none empty), its sums that split's part of `y` (a
    // last step short of 32 where `k` is: f16 weights')
    let all = (p.k + 31u) / 32u;
    let per = (all + p.splits - 1u) / p.splits;
    let s0 = wg.z * per;
    let s1 = min(all, s0 + per);
    let zo = wg.z * p.m * p.n;
    // (the tokens as [`X_F16_TILED`] gives them: a step's for every padded token together)
    let padded = ((p.m + 127u) / 128u) * 128u;
    let xo = (t0 + lr) * 8u + lh * 4u;
    let xs = padded * 8u;
    var c00 = coop_mat16x16<f32, C>();
    var c01 = coop_mat16x16<f32, C>();
    var c02 = coop_mat16x16<f32, C>();
    var c03 = coop_mat16x16<f32, C>();
    var c10 = coop_mat16x16<f32, C>();
    var c11 = coop_mat16x16<f32, C>();
    var c12 = coop_mat16x16<f32, C>();
    var c13 = coop_mat16x16<f32, C>();
    // the multiply-adds' sums in f16 (twice as fast), folded into the f32 ones every FOLD steps
    var h00 = coop_mat16x16<f16, C>();
    var h01 = coop_mat16x16<f16, C>();
    var h02 = coop_mat16x16<f16, C>();
    var h03 = coop_mat16x16<f16, C>();
    var h10 = coop_mat16x16<f16, C>();
    var h11 = coop_mat16x16<f16, C>();
    var h12 = coop_mat16x16<f16, C>();
    var h13 = coop_mat16x16<f16, C>();
    // the identity: an f16 sum's way into its f32 one (staged as an A, times it)
    if (li < 64u) {
        let col = li / 4u;
        var v = vec4<f16>(0.0h);
        if (col / 4u == li % 4u) { v[col % 4u] = 1.0h; }
        wt[li] = v;
    }
    workgroupBarrier();
    let i0 = 0u;
    let s4i = 4u;
    let ident = coopLoad<coop_mat16x16<f16, B>>(&wt[i0], s4i);
    workgroupBarrier();
    // the words of a step's weights and tokens this thread decodes and copies, loaded a step ahead
DECODE_REGS
    var xr0 = vec4<f16>();
    var xr1 = vec4<f16>();
    var xr2 = vec4<f16>();
    var xr3 = vec4<f16>();
    // the first step's; then each step's next loaded before it multiplies (the loads in flight as it does) and stored
    // after
    {
        let b = s0;
        LOAD_BLOCK
        STEP_LOAD
X_LOAD
    }
    {
        let b = s0;
        let buf = (s0 % 2u) * BUF4;
        DECODE_STEP
        let xa = buf + lr * S4 + lh * 4u;
        xt[xa] = xr0;
        xt[xa + 1u] = xr1;
        xt[xa + 2u] = xr2;
        xt[xa + 3u] = xr3;
    }
    workgroupBarrier();
    // the steps in windows of FOLD, each window's f16 sums folded into the f32 ones after it (a loop in a loop: naga
    // wants a cooperative op's control flow uniform, and an `if` on the step is not to it)
    for (var w0 = s0; w0 < s1; w0 += FOLDu) {
    let w1 = min(w0 + FOLDu, s1);
    for (var b0 = w0; b0 < w1; b0++) {
        // the next step (the last's own again, stored where no one reads it after): loaded, multiplied, stored, with
        // no branch between (a branch lets the compiler sink the loads to their stores, past the multiplies)
        let b = min(b0 + 1u, s1 - 1u);
        let buf = ((b0 + 1u) % 2u) * BUF4;
        if (b % 8u == 0u && b != b0) {
            LOAD_BLOCK
        }
        STEP_LOAD
X_LOAD
        let cur = (b0 % 2u) * BUF4;
        // (every index and stride a `let` of its own: naga's SPIR-V wants a cooperative load's and store's operands
        // emitted before them); k's two halves of 16 written out
        {
            let kk = 0u;
            let s10 = S4;
            let ia0 = cur + sr * S4 + kk / 4u;
            let ia1 = cur + (sr + 16u) * S4 + kk / 4u;
            let ib0 = cur + st * S4 + kk / 4u;
            let ib1 = ib0 + 16u * S4;
            let ib2 = ib0 + 32u * S4;
            let ib3 = ib0 + 48u * S4;
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia0], s10);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia1], s10);
            let b0f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib0], s10);
            let b1f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib1], s10);
            let b2f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib2], s10);
            let b3f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib3], s10);
            h00 = coopMultiplyAdd(a0, b0f, h00);
            h01 = coopMultiplyAdd(a0, b1f, h01);
            h02 = coopMultiplyAdd(a0, b2f, h02);
            h03 = coopMultiplyAdd(a0, b3f, h03);
            h10 = coopMultiplyAdd(a1, b0f, h10);
            h11 = coopMultiplyAdd(a1, b1f, h11);
            h12 = coopMultiplyAdd(a1, b2f, h12);
            h13 = coopMultiplyAdd(a1, b3f, h13);
        }
        {
            let kk = 16u;
            let s10 = S4;
            let ia0 = cur + sr * S4 + kk / 4u;
            let ia1 = cur + (sr + 16u) * S4 + kk / 4u;
            let ib0 = cur + st * S4 + kk / 4u;
            let ib1 = ib0 + 16u * S4;
            let ib2 = ib0 + 32u * S4;
            let ib3 = ib0 + 48u * S4;
            let a0 = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia0], s10);
            let a1 = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia1], s10);
            let b0f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib0], s10);
            let b1f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib1], s10);
            let b2f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib2], s10);
            let b3f = coopLoad<coop_mat16x16<f16, B>>(&xt[ib3], s10);
            h00 = coopMultiplyAdd(a0, b0f, h00);
            h01 = coopMultiplyAdd(a0, b1f, h01);
            h02 = coopMultiplyAdd(a0, b2f, h02);
            h03 = coopMultiplyAdd(a0, b3f, h03);
            h10 = coopMultiplyAdd(a1, b0f, h10);
            h11 = coopMultiplyAdd(a1, b1f, h11);
            h12 = coopMultiplyAdd(a1, b2f, h12);
            h13 = coopMultiplyAdd(a1, b3f, h13);
        }
        DECODE_STEP
        let xa = buf + lr * S4 + lh * 4u;
        xt[xa] = xr0;
        xt[xa + 1u] = xr1;
        xt[xa + 2u] = xr2;
        xt[xa + 3u] = xr3;
        workgroupBarrier();
    }
    {
        let cur = ((w1 - 1u) % 2u) * BUF4;
        // the f16 sums into the f32 ones, four fragments at a time through the step's buffer (two in its
        // weights', two in its tokens'), the f16 ones started over
        let fw0 = cur + sg * 128u;
        let fw1 = fw0 + 64u;
        let s4f = 4u;
        coopStore(h00, &wt[fw0], s4f);
        coopStore(h01, &wt[fw1], s4f);
        coopStore(h02, &xt[fw0], s4f);
        coopStore(h03, &xt[fw1], s4f);
        workgroupBarrier();
        let g00 = coopLoad<coop_mat16x16<f16, A>>(&wt[fw0], s4f);
        c00 = coopMultiplyAdd(g00, ident, c00);
        let g01 = coopLoad<coop_mat16x16<f16, A>>(&wt[fw1], s4f);
        c01 = coopMultiplyAdd(g01, ident, c01);
        let g02 = coopLoad<coop_mat16x16<f16, A>>(&xt[fw0], s4f);
        c02 = coopMultiplyAdd(g02, ident, c02);
        let g03 = coopLoad<coop_mat16x16<f16, A>>(&xt[fw1], s4f);
        c03 = coopMultiplyAdd(g03, ident, c03);
        workgroupBarrier();
        coopStore(h10, &wt[fw0], s4f);
        coopStore(h11, &wt[fw1], s4f);
        coopStore(h12, &xt[fw0], s4f);
        coopStore(h13, &xt[fw1], s4f);
        workgroupBarrier();
        let g10 = coopLoad<coop_mat16x16<f16, A>>(&wt[fw0], s4f);
        c10 = coopMultiplyAdd(g10, ident, c10);
        let g11 = coopLoad<coop_mat16x16<f16, A>>(&wt[fw1], s4f);
        c11 = coopMultiplyAdd(g11, ident, c11);
        let g12 = coopLoad<coop_mat16x16<f16, A>>(&xt[fw0], s4f);
        c12 = coopMultiplyAdd(g12, ident, c12);
        let g13 = coopLoad<coop_mat16x16<f16, A>>(&xt[fw1], s4f);
        c13 = coopMultiplyAdd(g13, ident, c13);
        workgroupBarrier();
        h00 = coop_mat16x16<f16, C>();
        h01 = coop_mat16x16<f16, C>();
        h02 = coop_mat16x16<f16, C>();
        h03 = coop_mat16x16<f16, C>();
        h10 = coop_mat16x16<f16, C>();
        h11 = coop_mat16x16<f16, C>();
        h12 = coop_mat16x16<f16, C>();
        h13 = coop_mat16x16<f16, C>();

    }
    }
    // out: y[token, row] is the tile's (row, token) column-major, a token's rows `n` apart
    let ns = p.n;
    let full = r0 + 128u <= p.rows && t0 + 128u <= p.m;
    if (full) {
        let o = zo + (t0 + st) * ns + p.row0 + r0 + sr;
        let o01 = o + 16u * ns;
        let o02 = o + 32u * ns;
        let o03 = o + 48u * ns;
        let o10 = o + 16u;
        let o11 = o + 16u + 16u * ns;
        let o12 = o + 16u + 32u * ns;
        let o13 = o + 16u + 48u * ns;
        coopStore(c00, &y[o], ns);
        coopStore(c01, &y[o01], ns);
        coopStore(c02, &y[o02], ns);
        coopStore(c03, &y[o03], ns);
        coopStore(c10, &y[o10], ns);
        coopStore(c11, &y[o11], ns);
        coopStore(c12, &y[o12], ns);
        coopStore(c13, &y[o13], ns);
    } else {
        // a tile at the edge: each fragment through the subgroup's staging, its rows and tokens in bounds
        EDGE_STORES
    }
}
"#;

/// [`coop_tiled`]'s store of a fragment at a tile's edge.
const COOP_EDGE: &str = r#"        {
            let eo = sg * 256u;
            coopStore(CF, &edge[eo], 16u);
            workgroupBarrier();
            for (var e = lane; e < 256u; e += 32u) {
                let row = r0 + sr + FR + e % 16u;
                let t = t0 + st + FT + e / 16u;
                if (row < p.rows && t < p.m) { y[zo + t * p.n + p.row0 + row] = edge[sg * 256u + e]; }
            }
            workgroupBarrier();
        }
"#;

/// [`coop8_tiled`]'s kernel: [`COOP_KERNEL`]'s tiles and steps, its fragments Metal's 8 x 8 (an Apple GPU's simdgroup
/// matrices), f16 into f32 sums.
const COOP8_KERNEL: &str = r#"enable f16;
enable wgpu_cooperative_matrix;
struct Params { k: u32, n: u32, m: u32, row0: u32, rows: u32, row_bytes: u32, splits: u32, _pad1: u32, }
WEIGHTS_BINDING
@group(0) @binding(1) var<storage, read> x16: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// a step's weights [row][k] and tokens [token][k], as halves: Metal loads a fragment from an array of its scalar, a
// row S halves apart (the decodes' indices are vec4s', S4 apart, each write four halves: `wt_put`); a subgroup's
// fragment staged at a tile's edge. One step's, not COOP_KERNEL's two: two take 43 KB, past Metal's 32, so the next
// step is decoded once every subgroup has multiplied this one (a barrier more a step)
const S4: u32 = 10u;
const S: u32 = 40u;
var<workgroup> wt: array<f16, 5120>;
var<workgroup> xt: array<f16, 5120>;
var<workgroup> edge: array<f32, 512>;

fn wt_put(i: u32, v: vec4<f16>) {
    let o = 4u * i;
    wt[o] = v.x;
    wt[o + 1u] = v.y;
    wt[o + 2u] = v.z;
    wt[o + 3u] = v.w;
}

fn xt_put(i: u32, v: vec4<f16>) {
    let o = 4u * i;
    xt[o] = v.x;
    xt[o + 1u] = v.y;
    xt[o + 2u] = v.z;
    xt[o + 3u] = v.w;
}

DECODE_HELPERS

// a byte as f32, less `o`: its bits in 2^23's mantissa (exact, no conversion)
fn byte_less(w: u32, o: f32) -> vec4<f32> {
    let b = vec4<u32>(w & 255u, (w >> 8u) & 255u, (w >> 16u) & 255u, w >> 24u);
    return bitcast<vec4<f32>>(b | vec4<u32>(0x4b000000u)) - vec4<f32>(8388608.0 + o);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r0 = wg.x * 128u;
    let t0 = wg.y * 128u;
    let sg = li / 32u;
    let lane = li % 32u;
    // the subgroup's 32 rows by 64 tokens of the tile
    let sr = (sg % 4u) * 32u;
    let st = (sg / 4u) * 64u;
    // what this thread decodes and copies: weight row lr's half lh of a step's 32, and token row lr's
    let lr = li / 2u;
    let lh = li % 2u;
    let rr = r0 + lr;
    let rl = min(rr, p.rows - 1u);
    let kx = p.k;
    // the workgroup's split of the steps (`wg.z` of `p.splits`, none empty), its sums that split's part of `y`
    let all = (p.k + 31u) / 32u;
    let per = (all + p.splits - 1u) / p.splits;
    let s0 = wg.z * per;
    let s1 = min(all, s0 + per);
    let zo = wg.z * p.m * p.n;
    // (the tokens as [`X_F16_TILED`] gives them: a step's for every padded token together)
    let padded = ((p.m + 127u) / 128u) * 128u;
    let xo = (t0 + lr) * 8u + lh * 4u;
    let xs = padded * 8u;
    // the subgroup's 4 by 8 fragments' sums: weight rows 8 r.., tokens 8 t..
SUMS
    // the words of a step's weights and tokens this thread decodes and copies, loaded a step ahead
DECODE_REGS
    var xr0 = vec4<f16>();
    var xr1 = vec4<f16>();
    var xr2 = vec4<f16>();
    var xr3 = vec4<f16>();
    {
        let b = s0;
        LOAD_BLOCK
X_LOAD
    }
    let buf = 0u;
    let cur = 0u;
    {
        let b = s0;
        DECODE_STEP
        let xa = buf + lr * S4 + lh * 4u;
        xt_put(xa, xr0);
        xt_put(xa + 1u, xr1);
        xt_put(xa + 2u, xr2);
        xt_put(xa + 3u, xr3);
    }
    workgroupBarrier();
    for (var b0 = s0; b0 < s1; b0++) {
        // the next step's words (the last's own again, decoded where no one reads it after): loaded as this one is
        // multiplied, then decoded over it once every subgroup has
        let b = min(b0 + 1u, s1 - 1u);
        if (b % 8u == 0u && b != b0) {
            LOAD_BLOCK
        }
X_LOAD
        // (every index and stride a `let` of its own, as COOP_KERNEL's); k's four eights written out
MULTIPLY
        workgroupBarrier();
        DECODE_STEP
        let xa = buf + lr * S4 + lh * 4u;
        xt_put(xa, xr0);
        xt_put(xa + 1u, xr1);
        xt_put(xa + 2u, xr2);
        xt_put(xa + 3u, xr3);
        workgroupBarrier();
    }
    // out: y[token, row] is the tile's (row, token) column-major, a token's rows `n` apart
    let ns = p.n;
    let full = r0 + 128u <= p.rows && t0 + 128u <= p.m;
    if (full) {
        let o = zo + (t0 + st) * ns + p.row0 + r0 + sr;
FULL_STORES
    } else {
        // a tile at the edge: each fragment through the subgroup's staging, its rows and tokens in bounds
EDGE_STORES
    }
}
"#;

/// [`coop8_tiled`]'s store of a fragment at a tile's edge.
const COOP8_EDGE: &str = r#"        {
            let eo = sg * 64u;
            let es = 8u;
            coopStore(CF, &edge[eo], es);
            workgroupBarrier();
            for (var e = lane; e < 64u; e += 32u) {
                let row = r0 + sr + FR + e % 8u;
                let t = t0 + st + FT + e / 8u;
                if (row < p.rows && t < p.m) { y[zo + t * p.n + p.row0 + row] = edge[eo + e]; }
            }
            workgroupBarrier();
        }
"#;

/// Q3_K's decode for [`coop_tiled`]: its helpers, and a thread's half of a row's 32-block.
const COOP_Q3K_HELPERS: &str = r#"fn q3_bytes(qw: u32, hw: u32, j: u32, h: u32) -> u32 {
    return ((qw >> (2u * j)) & 0x03030303u) | (((hw >> (j + 4u * h)) & 0x01010101u) << 2u);
}

// Q3_K: the scales of run j's two halves of 16 (sub-blocks 8 h + 2 j and the next), less 32.
fn q3_scales32(s: vec4<u32>, h: u32, j: u32) -> vec2<f32> {
    let sa = ((s.x >> (4u * h)) & 0x0f0f0f0fu) | (((s.z >> (4u * h)) & 0x03030303u) << 4u);
    let sb = ((s.y >> (4u * h)) & 0x0f0f0f0fu) | (((s.z >> (4u * h + 2u)) & 0x03030303u) << 4u);
    let sw = select(sa, sb, j >= 2u);
    let bs = 16u * (j % 2u);
    return vec2<f32>(f32((sw >> bs) & 255u) - 32.0, f32((sw >> (bs + 8u)) & 255u) - 32.0);
}
"#;
/// The other K-quants' decodes for [`coop_tiled`]: Q4_K's and Q5_K's (their 6-bit scales and minimums), Q6_K's
/// (its 210-byte blocks read as words, two scales a 32-block).
const COOP_K_HELPERS: &str = r#"// Byte `b` (0..12) of a block's scales, the header's last three words.
fn sbyte(h: vec4<u32>, b: u32) -> u32 {
    let wd = select(select(h.w, h.z, b < 8u), h.y, b < 4u);
    return (wd >> (8u * (b % 4u))) & 255u;
}
// Sub-block `j`'s 6-bit scale and minimum, as ggml's get_scale_min_k4.
fn scale_min(h: vec4<u32>, j: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(sbyte(h, j) & 63u), f32(sbyte(h, j + 4u) & 63u));
    }
    let sc = (sbyte(h, j + 4u) & 15u) | ((sbyte(h, j - 4u) >> 6u) << 4u);
    let mn = (sbyte(h, j + 4u) >> 4u) | ((sbyte(h, j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(mn));
}
"#;
const COOP_Q4K_REGS: &str = r#"    var hdw = vec4<u32>();
    var q0w = vec4<u32>();
    var q1w = vec4<u32>();
    var q2w = vec4<u32>();
    var q3w = vec4<u32>();"#;
const COOP_Q4K_LOAD: &str = r#"let lb4 = rl * (p.row_bytes / 16u) + (b / 8u) * 9u;
        hdw = w4[lb4];
        q0w = w4[lb4 + 1u + lh];
        q1w = w4[lb4 + 3u + lh];
        q2w = w4[lb4 + 5u + lh];
        q3w = w4[lb4 + 7u + lh];"#;
const COOP_Q4K_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let sb = b % 8u;
            let sh = 4u * (sb % 2u);
            let pr = sb / 2u;
            let qw = select(select(q0w, q1w, pr == 1u), select(q2w, q3w, pr == 3u), pr >= 2u);
            let dm = unpack2x16float(hdw.x);
            let smn = scale_min(hdw, sb);
            let dsc = dm.x * smn.x;
            let dmn = vec4<f32>(dm.y * smn.y);
            wt[at4] = vec4<f16>(dsc * byte_less((qw.x >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 1u] = vec4<f16>(dsc * byte_less((qw.y >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 2u] = vec4<f16>(dsc * byte_less((qw.z >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
            wt[at4 + 3u] = vec4<f16>(dsc * byte_less((qw.w >> sh) & 0x0f0f0f0fu, 0.0) - dmn);
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;
const COOP_Q5K_REGS: &str = r#"    var hdw = vec4<u32>();
    var hw = vec4<u32>();
    var q0w = vec4<u32>();
    var q1w = vec4<u32>();
    var q2w = vec4<u32>();
    var q3w = vec4<u32>();"#;
const COOP_Q5K_LOAD: &str = r#"let lb4 = rl * (p.row_bytes / 16u) + (b / 8u) * 11u;
        hdw = w4[lb4];
        hw = w4[lb4 + 1u + lh];
        q0w = w4[lb4 + 3u + lh];
        q1w = w4[lb4 + 5u + lh];
        q2w = w4[lb4 + 7u + lh];
        q3w = w4[lb4 + 9u + lh];"#;
const COOP_Q5K_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let sb = b % 8u;
            let sh = 4u * (sb % 2u);
            let pr = sb / 2u;
            let qw = select(select(q0w, q1w, pr == 1u), select(q2w, q3w, pr == 3u), pr >= 2u);
            let dm = unpack2x16float(hdw.x);
            let smn = scale_min(hdw, sb);
            let dsc = dm.x * smn.x;
            let dmn = vec4<f32>(dm.y * smn.y);
            wt[at4] = vec4<f16>(dsc * byte_less(((qw.x >> sh) & 0x0f0f0f0fu) | (((hw.x >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 1u] = vec4<f16>(dsc * byte_less(((qw.y >> sh) & 0x0f0f0f0fu) | (((hw.y >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 2u] = vec4<f16>(dsc * byte_less(((qw.z >> sh) & 0x0f0f0f0fu) | (((hw.z >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
            wt[at4 + 3u] = vec4<f16>(dsc * byte_less(((qw.w >> sh) & 0x0f0f0f0fu) | (((hw.w >> sb) & 0x01010101u) << 4u), 0.0) - dmn);
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;
const COOP_Q6K_HELPERS: &str = r#"fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
// The four bytes at `o`, an even offset (a word, or the halves of two).
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}
"#;
const COOP_Q6K_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let sub = b % 8u;
            let h = sub / 4u;
            let qd = sub % 4u;
            let lshift = select(0u, 4u, qd >= 2u);
            let hshift = 2u * qd;
            let base = rr * p.row_bytes + (b / 8u) * 210u;
            let d = unpack2x16float(byte(base + 208u) | (byte(base + 209u) << 8u)).x;
            let ql = base + h * 64u + (qd & 1u) * 32u + lh * 16u;
            let qh = base + 128u + h * 32u + lh * 16u;
            let scb = byte(base + 192u + h * 8u + 2u * qd + lh);
            let sc = d * f32(i32(scb) - select(0, 256, scb >= 128u));
            for (var wi = 0u; wi < 4u; wi++) {
                let q = ((word_at(ql + 4u * wi) >> lshift) & 0x0f0f0f0fu) | (((word_at(qh + 4u * wi) >> hshift) & 0x03030303u) << 4u);
                wt[at4 + wi] = vec4<f16>(sc * byte_less(q, 32.0));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;
const COOP_Q3K_REGS: &str = r#"    var hmw = vec4<u32>();
    var qs0w = vec4<u32>();
    var qs1w = vec4<u32>();
    var sdw = vec4<u32>();"#;
const COOP_Q3K_LOAD: &str = r#"let lb4 = rl * (p.row_bytes / 16u) + (b / 8u) * 7u;
        hmw = w4[lb4 + lh];
        qs0w = w4[lb4 + 2u + lh];
        qs1w = w4[lb4 + 4u + lh];
        sdw = w4[lb4 + 6u];"#;
const COOP_Q3K_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let q = b % 8u;
            let h = q / 4u;
            let j = q % 4u;
            let scs = q3_scales32(sdw, h, j) * unpack2x16float(sdw.w & 0xffffu).x;
            let sc = select(scs.x, scs.y, lh == 1u);
            let qsw = select(qs0w, qs1w, h == 1u);
            wt[at4] = vec4<f16>(sc * byte_less(q3_bytes(qsw.x, hmw.x, j, h), 4.0));
            wt[at4 + 1u] = vec4<f16>(sc * byte_less(q3_bytes(qsw.y, hmw.y, j, h), 4.0));
            wt[at4 + 2u] = vec4<f16>(sc * byte_less(q3_bytes(qsw.z, hmw.z, j, h), 4.0));
            wt[at4 + 3u] = vec4<f16>(sc * byte_less(q3_bytes(qsw.w, hmw.w, j, h), 4.0));
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// Q8_0's decode for [`coop_tiled`] (its 34-byte blocks read as words, [`COOP_Q6K_HELPERS`]'): a step one block, a
/// thread's half of it its scale times its 16 int8s (each byte's sign bit flipped: its value plus 128).
const COOP_Q8_0_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let base = rr * p.row_bytes + b * 34u;
            let d = unpack2x16float(byte(base) | (byte(base + 1u) << 8u)).x;
            let q = base + 2u + lh * 16u;
            for (var wi = 0u; wi < 4u; wi++) {
                wt[at4 + wi] = vec4<f16>(d * byte_less(word_at(q + 4u * wi) ^ 0x80808080u, 128.0));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// Q2_0's decode for [`coop_tiled`] (18-byte blocks of 64 weights, two steps a block): a thread's half of a step its
/// scale times its 16 codes less one (four codes a byte; each product an f16 exactly).
const COOP_Q2_0_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let base = rr * p.row_bytes + (b / 2u) * 18u;
            let d = unpack2x16float(byte(base) | (byte(base + 1u) << 8u)).x;
            let codes = word_at(base + 2u + (b % 2u) * 8u + lh * 4u);
            for (var wi = 0u; wi < 4u; wi++) {
                let q = (codes >> (wi * 8u)) & 0xffu;
                wt[at4 + wi] = vec4<f16>(d * (vec4<f32>(f32(q & 3u), f32((q >> 2u) & 3u), f32((q >> 4u) & 3u), f32(q >> 6u)) - 1.0));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// Q4_0's decode for [`coop_tiled`] (on the GPU 20-byte blocks, [`padded_block`]: the scale, a gap, 16 bytes of
/// nibbles): a step a block, a thread's half its low nibbles or its high ones, each less 8.
const COOP_Q4_0_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let base = rr * p.row_bytes + b * 20u;
            let d = unpack2x16float(byte(base) | (byte(base + 1u) << 8u)).x;
            for (var wi = 0u; wi < 4u; wi++) {
                wt[at4 + wi] = vec4<f16>(d * byte_less((word_at(base + 4u + 4u * wi) >> (lh * 4u)) & 0x0f0f0f0fu, 8.0));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// Q5_0's decode for [`coop_tiled`] (22-byte blocks: the scale, 32 high bits, 16 bytes of nibbles): a step a block,
/// a thread's half its low nibbles or its high ones, each with its fifth bit, less 16.
const COOP_Q5_0_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let base = rr * p.row_bytes + b * 22u;
            let d = unpack2x16float(byte(base) | (byte(base + 1u) << 8u)).x;
            let qh = word_at(base + 2u) >> (lh * 16u);
            for (var wi = 0u; wi < 4u; wi++) {
                let hb = (qh >> (wi * 4u)) & 15u;
                let fifth = ((hb & 1u) | ((hb & 2u) << 7u) | ((hb & 4u) << 14u) | ((hb & 8u) << 21u)) << 4u;
                wt[at4 + wi] = vec4<f16>(d * byte_less(((word_at(base + 6u + 4u * wi) >> (lh * 4u)) & 0x0f0f0f0fu) | fifth, 16.0));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// [`COOP_Q6K_HELPERS`] and IQ4's values (a nibble's, ggml's `kvalues_iq4nl`), a word's four nibbles' at once.
const COOP_IQ4_HELPERS: &str = r#"fn byte(o: u32) -> u32 { return (w[o >> 2u] >> ((o & 3u) * 8u)) & 0xffu; }
fn word_at(o: u32) -> u32 {
    let i = o >> 2u;
    if ((o & 3u) == 0u) { return w[i]; }
    return (w[i] >> 16u) | (w[i + 1u] << 16u);
}
const KV4 = array<f32, 16>(-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
// the values of a word's four nibbles (each byte's low four bits)
fn iq4(q: u32) -> vec4<f32> {
    return vec4<f32>(KV4[q & 15u], KV4[(q >> 8u) & 15u], KV4[(q >> 16u) & 15u], KV4[(q >> 24u) & 15u]);
}
"#;

/// IQ4_NL's decode for [`coop_tiled`] (18-byte blocks: the scale, 16 bytes of nibbles): a step a block, a thread's
/// half its low nibbles' values or its high ones'.
const COOP_IQ4_NL_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let base = rr * p.row_bytes + b * 18u;
            let d = unpack2x16float(byte(base) | (byte(base + 1u) << 8u)).x;
            for (var wi = 0u; wi < 4u; wi++) {
                wt[at4 + wi] = vec4<f16>(d * iq4(word_at(base + 2u + 4u * wi) >> (lh * 4u)));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// IQ4_XS's decode for [`coop_tiled`] (136-byte blocks of 256: the scale, each 32's six-bit scale in two fields, 128
/// bytes of nibbles): a step a 32 of its block, its scale the block's times its own less 32.
const COOP_IQ4_XS_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let ib = b % 8u;
            let base = rr * p.row_bytes + (b / 8u) * 136u;
            let head = w[base >> 2u];
            let low = (w[(base >> 2u) + 1u] >> (4u * ib)) & 15u;
            let ls = low | ((((head >> 16u) >> (2u * ib)) & 3u) << 4u);
            let dl = unpack2x16float(head & 0xffffu).x * (f32(ls) - 32.0);
            let qo = (base >> 2u) + 2u + ib * 4u;
            for (var wi = 0u; wi < 4u; wi++) {
                wt[at4 + wi] = vec4<f16>(dl * iq4(w[qo + wi] >> (lh * 4u)));
            }
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// f16 weights for [`coop_tiled`] (`[n, k]`, `k` of 4): a thread's 16 of its row's step loaded a step ahead as the
/// tokens' are, and stored as they are (a last step's past `k` zeros).
const COOP_F16_REGS: &str = r#"    var wr0 = vec4<f16>();
    var wr1 = vec4<f16>();
    var wr2 = vec4<f16>();
    var wr3 = vec4<f16>();"#;
const COOP_F16_LOAD: &str = r#"let wk = b * 32u + lh * 16u;
        let wo = rl * (kx / 4u) + wk / 4u;
        wr0 = select(vec4<f16>(), w4[wo], wk < kx);
        wr1 = select(vec4<f16>(), w4[wo + 1u], wk + 4u < kx);
        wr2 = select(vec4<f16>(), w4[wo + 2u], wk + 8u < kx);
        wr3 = select(vec4<f16>(), w4[wo + 3u], wk + 12u < kx);"#;
const COOP_F16_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            wt[at4] = wr0;
            wt[at4 + 1u] = wr1;
            wt[at4 + 2u] = wr2;
            wt[at4 + 3u] = wr3;
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// [`COOP_KERNEL`]'s tokens' loads of a step `b`: a thread's 16 of its token's 32 from the tiled copy
/// ([`X_F16_TILED`]).
const COOP_X_TILED: &str = r#"        let xb = xo + b * xs;
        xr0 = x16[xb];
        xr1 = x16[xb + 1u];
        xr2 = x16[xb + 2u];
        xr3 = x16[xb + 3u];"#;

/// A 3x3 convolution's tokens' loads for [`coop_conv3x3`]: the tokens an image's pixels (`p.m` of them, rows of
/// `p.row_bytes`, `p._pad1` rows), its input `x16` f16 channels-last with each pixel's channels padded to 32 (`p.k / 9`
/// of them); step `b` tap `b / cs` (`3 dy + dx`) and channels `32 (b % cs)..` of it, a thread's 16 of them from the
/// pixel `(y + dy - 1, x + dx - 1)`, zeros past the image's edge.
const COOP_X_CONV3X3: &str = r#"        let ctap = b / cs;
        let cpix = min(t0 + lr, p.m - 1u);
        let cy = cpix / p.row_bytes + ctap / 3u;
        let cx = cpix % p.row_bytes + ctap % 3u;
        let cin = cy >= 1u && cy <= p._pad1 && cx >= 1u && cx <= p.row_bytes;
        let cb = ((select(0u, cy - 1u, cin) * p.row_bytes + select(0u, cx - 1u, cin)) * cs * 32u + (b % cs) * 32u + lh * 16u) / 4u;
        xr0 = select(vec4<f16>(), x16[cb], cin);
        xr1 = select(vec4<f16>(), x16[cb + 1u], cin);
        xr2 = select(vec4<f16>(), x16[cb + 2u], cin);
        xr3 = select(vec4<f16>(), x16[cb + 3u], cin);"#;

/// A 3x3x3 convolution's tokens' loads for [`coop_conv`]: as [`COOP_X_CONV3X3`]'s over a video's voxels (frames of
/// `p._pad1` rows of `p.row_bytes`), tap `ctap` (`9 dt + 3 dy + dx`) from frame `t + dt - 1` clamped to the clip (its
/// first and last repeated past its ends) and the pixel `(y + dy - 1, x + dx - 1)`, zeros past the frame's edge.
const COOP_X_CONV3D: &str = r#"        let ctap = b / cs;
        let cpix = min(t0 + lr, p.m - 1u);
        let plane = p.row_bytes * p._pad1;
        let cfr = cpix / plane;
        let rem = cpix % plane;
        let it = u32(clamp(i32(cfr) + i32(ctap / 9u) - 1, 0, i32(p.m / plane) - 1));
        let cy = rem / p.row_bytes + (ctap / 3u) % 3u;
        let cx = rem % p.row_bytes + ctap % 3u;
        let cin = cy >= 1u && cy <= p._pad1 && cx >= 1u && cx <= p.row_bytes;
        let cb = ((it * plane + select(0u, (cy - 1u) * p.row_bytes + cx - 1u, cin)) * cs * 32u + (b % cs) * 32u + lh * 16u) / 4u;
        xr0 = select(vec4<f16>(), x16[cb], cin);
        xr1 = select(vec4<f16>(), x16[cb + 1u], cin);
        xr2 = select(vec4<f16>(), x16[cb + 2u], cin);
        xr3 = select(vec4<f16>(), x16[cb + 3u], cin);"#;

/// A 1x1 convolution's tokens' loads for [`coop_conv`]: as [`COOP_X_CONV3X3`]'s with one tap, the pixel's own.
const COOP_X_CONV1X1: &str = r#"        let cpix = min(t0 + lr, p.m - 1u);
        let cb = (cpix * cs * 32u + (b % cs) * 32u + lh * 16u) / 4u;
        xr0 = x16[cb];
        xr1 = x16[cb + 1u];
        xr2 = x16[cb + 2u];
        xr3 = x16[cb + 3u];"#;

/// NVFP4 for [`coop_tiled_nvfp4`]: a row's words its nibbles (`k / 8`) then its block scales (`k / 64`, four E4M3 a
/// word), `p.row_bytes` words a row; E2M1 and E4M3 made from their bits (no table), each value times its block's
/// scale exact in f16 (6 bits, within 2^-10..2688), the tensor's own scale after the sums.
const COOP_NVFP4_HELPERS: &str = r#"fn e2m1(c: u32) -> f32 {
    let e = (c >> 1u) & 3u;
    let normal = ((e + 126u) << 23u) | ((c & 1u) << 22u);
    let sub = select(0u, 0x3f000000u, (c & 1u) != 0u);
    return bitcast<f32>(((c & 8u) << 28u) | select(normal, sub, e == 0u));
}

fn e4m3(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let v = select(bitcast<f32>(((e + 120u) << 23u) | (m << 20u)), f32(m) * 0.001953125, e == 0u);
    return select(v, -v, (b & 128u) != 0u);
}

// a word's eight values, high nibble of each byte first
fn nib4(w: u32, sh: u32, s: f32) -> vec4<f16> {
    let q = w >> sh;
    return vec4<f16>(s * vec4<f32>(e2m1((q >> 4u) & 15u), e2m1(q & 15u), e2m1((q >> 12u) & 15u), e2m1((q >> 8u) & 15u)));
}"#;
const COOP_NVFP4_REGS: &str = r#"    var nw0 = 0u;
    var nw1 = 0u;
    var sw = 0u;"#;
const COOP_NVFP4_LOAD: &str = r#"let nb = rl * p.row_bytes + b * 4u + lh * 2u;
        nw0 = w[nb];
        nw1 = w[nb + 1u];
        sw = w[rl * p.row_bytes + kx / 8u + (2u * b + lh) / 4u];"#;
const COOP_NVFP4_STEP: &str = r#"let at4 = buf + lr * S4 + lh * 4u;
        if (rr < p.rows) {
            let sc = e4m3((sw >> (8u * ((2u * b + lh) % 4u))) & 0xffu);
            wt[at4] = nib4(nw0, 0u, sc);
            wt[at4 + 1u] = nib4(nw0, 16u, sc);
            wt[at4 + 2u] = nib4(nw1, 0u, sc);
            wt[at4 + 3u] = nib4(nw1, 16u, sc);
        } else {
            for (var i = 0u; i < 4u; i++) { wt[at4 + i] = vec4<f16>(0.0h); }
        }"#;

/// [`coop_tiled`] for NVFP4 weights ([`COOP_NVFP4_HELPERS`]'s layout; `k` of 64), its sums f32 throughout.
pub fn coop_tiled_nvfp4() -> String {
    f32_sums(&coop_source("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_NVFP4_HELPERS, COOP_NVFP4_REGS, "", COOP_NVFP4_LOAD, COOP_NVFP4_STEP, COOP_X_TILED))
}

/// A prompt's matmul on the tensor cores (WGSL's cooperative matrices, f16 into f32): a workgroup a tile of
/// [`COOP_TILE`] weight rows by as many tokens, `k` 32 at a time; each step the tile's weights decoded to f16 and its
/// tokens' rows (as [`X_F16_TILED`] gives them, padded to the tile) copied into the workgroup's memory, the next step's
/// loaded as this one's are multiplied, each of its 8 subgroups its 32 rows by 64 tokens as 2 by 4 fragments of 16x16
/// (the tokens' fragments read from their rows in memory where they were the f16 rows in place: 141 TFLOPS without a
/// decode, the loop 207 with both in the workgroup's); the sums stored straight into `y` (a tile at the edge through a
/// fragment's staging). None for a type without one.
pub fn coop_tiled(dtype: GgmlType) -> Option<String> {
    let (binding, helpers, regs, load, step) = coop_parts_of(dtype)?;
    Some(coop_source(binding, helpers, regs, load, "", step, COOP_X_TILED))
}

/// A type's parts of [`coop_tiled`] and [`coop8_tiled`]: its weights' binding, helpers, registers, a block's loads and
/// a step's decode. None for a type without them.
fn coop_parts_of(dtype: GgmlType) -> Option<(&'static str, &'static str, &'static str, &'static str, &'static str)> {
    let vec4s = "@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;";
    // (Q6_K's words, 210-byte blocks two bytes off in every other, loaded as they are decoded)
    Some(match dtype {
        GgmlType::Q3_K => (vec4s, COOP_Q3K_HELPERS, COOP_Q3K_REGS, COOP_Q3K_LOAD, COOP_Q3K_STEP),
        GgmlType::Q4_K => (vec4s, COOP_K_HELPERS, COOP_Q4K_REGS, COOP_Q4K_LOAD, COOP_Q4K_STEP),
        GgmlType::Q5_K => (vec4s, COOP_K_HELPERS, COOP_Q5K_REGS, COOP_Q5K_LOAD, COOP_Q5K_STEP),
        GgmlType::Q6_K => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_Q6K_HELPERS, "", "", COOP_Q6K_STEP),
        GgmlType::Q8_0 => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_Q6K_HELPERS, "", "", COOP_Q8_0_STEP),
        // (the plain blocks a Qwen3.8-Flash-Next GGUF's dense matrices come in beside the K-quants: their bytes read
        // as Q6_K's and Q8_0's are, a step a 32 of a block)
        GgmlType::Q2_0 => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_Q6K_HELPERS, "", "", COOP_Q2_0_STEP),
        GgmlType::Q4_0 => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_Q6K_HELPERS, "", "", COOP_Q4_0_STEP),
        GgmlType::Q5_0 => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_Q6K_HELPERS, "", "", COOP_Q5_0_STEP),
        GgmlType::IQ4_NL => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_IQ4_HELPERS, "", "", COOP_IQ4_NL_STEP),
        GgmlType::IQ4_XS => ("@group(0) @binding(0) var<storage, read> w: array<u32>;", COOP_IQ4_HELPERS, "", "", COOP_IQ4_XS_STEP),
        _ => return None,
    })
}

/// [`coop_tiled`] for Metal, whose cooperative matrices (an Apple GPU's simdgroup matrices) are 8 x 8: the same tiles of
/// 128 weight rows by 128 tokens, the same steps of 32 of `k` decoded and loaded a step ahead, each subgroup's 32 rows by
/// 64 tokens as 4 by 8 fragments, f16 into f32 sums (no f16 sums to fold). None for a type without one.
///
/// Metal loads a fragment from an array of its scalar, a row its stride of them apart: the tiles are halves here, and
/// each type's decode, written for vec4s, writes four of them at a time (`wt_put`).
pub fn coop8_tiled(dtype: GgmlType) -> Option<String> {
    let (binding, helpers, regs, load, step) = coop_parts_of(dtype)?;
    Some(coop8_source(binding, helpers, regs, load, step))
}

/// [`COOP8_KERNEL`] with a type's weights put in, and its fragments' sums, multiply-adds and stores written out.
fn coop8_source(binding: &str, helpers: &str, regs: &str, load: &str, step: &str) -> String {
    let frags: Vec<(u32, u32)> = (0..4).flat_map(|r| (0..8).map(move |t| (r, t))).collect();
    let sums: String = frags.iter().map(|(r, t)| format!("    var c{r}{t} = coop_mat8x8<f32, C>();\n")).collect();
    let mut multiply = String::new();
    for kk in [0u32, 8, 16, 24] {
        multiply += &format!("        {{\n            let kk = {kk}u;\n            let s = S;\n");
        for r in 0..4 {
            multiply += &format!("            let ia{r} = 4u * cur + (sr + {}u) * S + kk;\n", 8 * r);
        }
        for t in 0..8 {
            multiply += &format!("            let ib{t} = 4u * cur + (st + {}u) * S + kk;\n", 8 * t);
        }
        for r in 0..4 {
            multiply += &format!("            let a{r} = coopLoadT<coop_mat8x8<f16, A>>(&wt[ia{r}], s);\n");
        }
        for t in 0..8 {
            multiply += &format!("            let b{t}f = coopLoad<coop_mat8x8<f16, B>>(&xt[ib{t}], s);\n");
        }
        for (r, t) in &frags {
            multiply += &format!("            c{r}{t} = coopMultiplyAdd(a{r}, b{t}f, c{r}{t});\n");
        }
        multiply += "        }\n";
    }
    let full: String = frags.iter().map(|(r, t)| format!("        let o{r}{t} = o + {}u * ns + {}u;\n        coopStore(c{r}{t}, &y[o{r}{t}], ns);\n", 8 * t, 8 * r)).collect();
    let edges: String = frags.iter().map(|(r, t)| COOP8_EDGE.replace("CF", &format!("c{r}{t}")).replace("FR", &format!("{}u", 8 * r)).replace("FT", &format!("{}u", 8 * t))).collect();
    COOP8_KERNEL
        .replace("WEIGHTS_BINDING", binding)
        .replace("DECODE_HELPERS", helpers)
        .replace("DECODE_REGS", regs)
        .replace("LOAD_BLOCK", load)
        .replace("X_LOAD", COOP_X_TILED)
        .replace("DECODE_STEP", &tile_writes_as_halves(step))
        .replace("SUMS\n", &sums)
        .replace("MULTIPLY\n", &multiply)
        .replace("FULL_STORES\n", &full)
        .replace("EDGE_STORES\n", &edges)
}

/// A type's decode with each write of four of its weights (`wt[i] = v;`, a vec4 of the tile) as [`COOP8_KERNEL`]'s
/// `wt_put(i, v);`, the tile being halves there.
fn tile_writes_as_halves(step: &str) -> String {
    let mut out = String::with_capacity(step.len());
    let mut rest = step;
    while let Some(at) = rest.find("wt[") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 3..];
        // the index, to its bracket; then ` = ` and the value, to its `;`
        let mut depth = 1;
        let close = after.char_indices().find(|&(_, c)| {
            depth += match c { '[' => 1, ']' => -1, _ => 0 };
            depth == 0
        }).expect("a write's index ends").0;
        let index = &after[..close];
        let value_start = after[close..].strip_prefix("] = ").expect("a decode's tile is written, `wt[i] = v;`");
        let end = value_start.find(';').expect("a write ends");
        out.push_str(&format!("wt_put({index}, {})", &value_start[..end]));
        rest = &value_start[end..];
    }
    out.push_str(rest);
    out
}

/// [`coop_tiled`] for f16 weights (`[n, k]` two to a word, `k` of 4: a last step short of 32 padded with zeros), its
/// sums f32 throughout: its matmuls (Qwen3.8-Flash-Next's hyper-connections', routers') are few multiply-adds for the
/// bytes they read, and f16 windows of 32 steps took a chained prompt's logits from 0.9986 of the host's (cosine) to
/// 0.9967 (0.9989 in f32).
pub fn coop_tiled_f16() -> String {
    f32_sums(&coop_source("@group(0) @binding(0) var<storage, read> w4: array<vec4<f16>>;", "", COOP_F16_REGS, "", COOP_F16_LOAD, COOP_F16_STEP, COOP_X_TILED))
}

/// A `taps` (1 or 9) convolution (stride 1, padding 1 for 3x3) on the tensor cores: [`coop_tiled_f16`]'s kernel with
/// its tokens an image's pixels, their rows gathered as [`COOP_X_CONV3X3`] (an implicit im2col) or
/// [`COOP_X_CONV1X1`] reads them; the weights `[cout, taps cin_p]` f16, each output's taps in turn, a tap's channels
/// padded to `cin_p` (32's). `Params`: k `taps cin_p`, n `cout`, m the pixels, rows `cout`, `row_bytes` the image's
/// width, `_pad1` its height.
pub fn coop_conv(taps: usize) -> String {
    let seven = COOP_X_CONV3X3
        .replace("ctap / 3u", "ctap / 7u")
        .replace("ctap % 3u", "ctap % 7u")
        .replace("cy >= 1u && cy <= p._pad1 && cx >= 1u && cx <= p.row_bytes", "cy >= 3u && cy < p._pad1 + 3u && cx >= 3u && cx < p.row_bytes + 3u")
        .replace("cy - 1u", "cy - 3u")
        .replace("cx - 1u", "cx - 3u");
    let (x_load, per) = match taps {
        27 => (COOP_X_CONV3D, 864),
        // (a 7x7's: the 3x3's with its taps 7 a row and 3 pixels off the edge)
        49 => (seven.as_str(), 1568),
        9 => (COOP_X_CONV3X3, 288),
        _ => (COOP_X_CONV1X1, 32),
    };
    let regs = format!("{COOP_F16_REGS}\n    // the steps a tap\n    let cs = p.k / {per}u;");
    f32_sums(&coop_source("@group(0) @binding(0) var<storage, read> w4: array<vec4<f16>>;", "", &regs, "", COOP_F16_LOAD, COOP_F16_STEP, x_load))
}

/// A [`COOP_KERNEL`] source with its multiply-adds into the f32 sums themselves (no f16 windows, nothing folded).
fn f32_sums(src: &str) -> String {
    let mut out = src.to_string();
    for f in ["00", "01", "02", "03", "10", "11", "12", "13"] {
        let (a, b) = (&f[..1], &f[1..]);
        let from = format!("h{f} = coopMultiplyAdd(a{a}, b{b}f, h{f});");
        assert_eq!(out.matches(&from).count(), 2, "the kernel's multiply-adds into h{f}");
        out = out.replace(&from, &format!("c{f} = coopMultiplyAdd(a{a}, b{b}f, c{f});"));
    }
    let start = out.find("    {\n        let cur = ((w1 - 1u) % 2u) * BUF4;").expect("the fold");
    let end_mark = "        h13 = coop_mat16x16<f16, C>();\n\n    }\n";
    let end = out[start..].find(end_mark).expect("the fold's end") + start + end_mark.len();
    out.replace_range(start..end, "");
    out
}

/// [`COOP_KERNEL`] with a type's weights put in: their binding, helpers, registers, a block's loads (every 8 steps), a
/// step's (beside its tokens'), and a step's decode into the workgroup's memory.
fn coop_source(binding: &str, helpers: &str, regs: &str, load: &str, step_load: &str, step: &str, x_load: &str) -> String {
    let frags = [("c00", 0u32, 0u32), ("c01", 0, 16), ("c02", 0, 32), ("c03", 0, 48), ("c10", 16, 0), ("c11", 16, 16), ("c12", 16, 32), ("c13", 16, 48)];
    let edges: String = frags.iter().map(|(cf, fr, ft)| COOP_EDGE.replace("CF", cf).replace("FR", &format!("{fr}u")).replace("FT", &format!("{ft}u"))).collect();
    let fold = std::env::var("OAIY_COOP_FOLD").ok().and_then(|v| v.parse::<u32>().ok()).filter(|&f| f > 0).unwrap_or(COOP_FOLD);
    COOP_KERNEL
        .replace("FOLDu", &format!("{fold}u"))
        .replace("WEIGHTS_BINDING", binding)
        .replace("DECODE_HELPERS", helpers)
        .replace("DECODE_REGS", regs)
        .replace("LOAD_BLOCK", load)
        .replace("STEP_LOAD", step_load)
        .replace("X_LOAD", x_load)
        .replace("DECODE_STEP", step)
        .replace("EDGE_STORES", &edges)
}

/// [`coop_tiled`] with its in-loop decode between marker comments (a measurement takes it out).
#[cfg(test)]
pub(crate) fn coop_tiled_marked(dtype: GgmlType) -> Option<String> {
    let src = coop_tiled(dtype)?;
    // the loop's decode is the second DECODE_STEP's: mark it in the template and fill it in again
    let _ = src;
    let (binding, helpers, regs, load, step) = match dtype {
        GgmlType::Q3_K => ("@group(0) @binding(0) var<storage, read> w4: array<vec4<u32>>;", COOP_Q3K_HELPERS, COOP_Q3K_REGS, COOP_Q3K_LOAD, COOP_Q3K_STEP),
        _ => return None,
    };
    let frags = [("c00", 0u32, 0u32), ("c01", 0, 16), ("c02", 0, 32), ("c03", 0, 48), ("c10", 16, 0), ("c11", 16, 16), ("c12", 16, 32), ("c13", 16, 48)];
    let edges: String = frags.iter().map(|(cf, fr, ft)| COOP_EDGE.replace("CF", cf).replace("FR", &format!("{fr}u")).replace("FT", &format!("{ft}u"))).collect();
    let marked = COOP_KERNEL.replacen("        DECODE_STEP\n        let xa = buf + lr * S4 + lh * 4u;\n        xt[xa] = xr0;", "SECOND_STEP", 2);
    // (the first is the prologue's, kept; the second the loop's, marked)
    let marked = marked.replacen("SECOND_STEP", "        DECODE_STEP\n        let xa = buf + lr * S4 + lh * 4u;\n        xt[xa] = xr0;", 1);
    let marked = marked.replacen("SECOND_STEP", "        // DECODE BEGIN\n        DECODE_STEP\n        // DECODE END\n        let xa = buf + lr * S4 + lh * 4u;\n        xt[xa] = xr0;", 1);
    Some(marked.replace("FOLDu", &format!("{COOP_FOLD}u")).replace("WEIGHTS_BINDING", binding).replace("DECODE_HELPERS", helpers).replace("DECODE_REGS", regs).replace("LOAD_BLOCK", load).replace("STEP_LOAD", "").replace("X_LOAD", COOP_X_TILED).replace("DECODE_STEP", step).replace("EDGE_STORES", &edges))
}
