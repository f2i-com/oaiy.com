//! The chain's kernels: the transforms before and after a matmul, and the matmul of a job a row, of a prompt's
//! blocks (plain and by tensor cores) and of a check's few rows, with the names their pipelines go by.

use super::*;

/// f32 to f16 and back as the host's `half` (the `half` crate's: to nearest even, subnormals, infinity past 65504),
/// for the chain's transforms: an activation is not always a normal f16 value, as a decoded weight is.
pub(crate) const HALF: &str = r#"
fn half(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    let sign = b & 0x80000000u;
    let a = b & 0x7fffffffu;
    if (a > 0x7f800000u) { return v; }
    if (a >= 0x477ff000u) { return bitcast<f32>(sign | 0x7f800000u); }
    if (a < 0x38800000u) {
        // below f16's normals: a multiple of 2^-24, to nearest even
        let q = round(bitcast<f32>(a) * 16777216.0) / 16777216.0;
        return bitcast<f32>(sign | bitcast<u32>(q));
    }
    return bitcast<f32>(sign | ((a + 0xfffu + ((a >> 13u) & 1u)) & 0xffffe000u));
}
"#;

/// The chain's EXL3 kernels take a job list: job `j` is matrix `jobs[2j]` of a group on row `jobs[2j + 1]` of its
/// input, its result row `j`. A projection's input transform for each job, a workgroup a (128-block, job): `x`'s row
/// gathered through the input map (unless `p[0].y`, the identity), rounded to f16 and scaled by `suh`, the Hadamard
/// transform of each 128-block, scaled by 1/sqrt(128) and rounded (as `Transform::pre`). `p[0]`: k, identity.
const G_PRE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> suh: array<f32>;
@group(0) @binding(2) var<storage, read> imap: array<u32>;
@group(0) @binding(3) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> xh: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = p[0].x;
    let j = wg.y;
    let m = jobs[2u * j];
    let xr = jobs[2u * j + 1u];
    let i = wg.x * 128u + t;
    var src = i;
    if (p[0].y == 0u) { src = imap[m * k + i]; }
    sh[t] = half(x[xr * k + src]) * suh[m * k + i];
    workgroupBarrier();
    for (var s = 1u; s < 128u; s *= 2u) {
        if ((t & s) == 0u) {
            let a = sh[t];
            let b = sh[t + s];
            sh[t] = a + b;
            sh[t + s] = a - b;
        }
        workgroupBarrier();
    }
    xh[j * k + i] = half(sh[t] * bitcast<f32>(0x3db504f4u));
}
"#;

/// [`G_PRE`] of a SwiGLU's output computed as it is read: job `j`'s row `xr` is `silu(g) * u`, `g` row `p[0].z xr +
/// p[0].w` of `x` and `u` row `p[1].x xr + p[1].y` of `up` (an expert group's gate and up rows `2 xr` and `2 xr + 1` of
/// one vector; a shared expert's row `xr` of two), as the SwiGLU kernels compute it: a dispatch fewer.
const G_PRE_SWIGLU: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> suh: array<f32>;
@group(0) @binding(2) var<storage, read> imap: array<u32>;
@group(0) @binding(3) var<storage, read> jobs: array<u32>;
@group(0) @binding(4) var<storage, read> up: array<f32>;
@group(0) @binding(6) var<storage, read_write> xh: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = p[0].x;
    let j = wg.y;
    let m = jobs[2u * j];
    let xr = jobs[2u * j + 1u];
    let i = wg.x * 128u + t;
    var src = i;
    if (p[0].y == 0u) { src = imap[m * k + i]; }
    let g = x[(p[0].z * xr + p[0].w) * k + src];
    let u = up[(p[1].x * xr + p[1].y) * k + src];
    sh[t] = half((g / (1.0 + exp(-g))) * u) * suh[m * k + i];
    workgroupBarrier();
    for (var s = 1u; s < 128u; s *= 2u) {
        if ((t & s) == 0u) {
            let a = sh[t];
            let b = sh[t + s];
            sh[t] = a + b;
            sh[t + s] = a - b;
        }
        workgroupBarrier();
    }
    xh[j * k + i] = half(sh[t] * bitcast<f32>(0x3db504f4u));
}
"#;

/// The matmul of each job's transformed row (the host's one-row kernel's, [`one_source`]; the matrices a group's:
/// matrix `m`'s words from `m * p[1].x`): a workgroup a (tile column, job and split), its partial sums to `part[(j *
/// splits + s) * n..]`, the jobs from `p[1].y` (a pass of a long list: 65535 workgroups an axis). `p[0]`: n, k, tile
/// words, splits; `p[1]`: words a matrix, the pass's first job.
fn g_mm_source() -> String {
    g_mm_source_with(None)
}

/// [`g_mm_source`] for tiles of `tw` words alone ([`lanes_fixed`]).
fn g_mm_fixed_source(tw: usize) -> String {
    g_mm_source_with(Some(tw))
}

fn g_mm_source_with(fixed: Option<usize>) -> String {
    let body = one_lanes_with("j * k + kt * 16u + {r}", "base + (kt * ntiles + nt) * nw", "", fixed);
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<private> j: u32;
var<private> k: u32;
{body}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    j = p[1].y + wg.z / splits;
    let s = wg.z % splits;
    let base = jobs[2u * j] * p[1].x;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    red[t] = lanes(t, nt, ntiles, ks, ke, tw, base);
    workgroupBarrier();
    if (t < 16u) {{
        part[(j * splits + s) * n + nt * 16u + t] = column(t);
    }}
}}
"#
    )
}

/// [`MANY`]'s matmul in a chain, for a prompt's rows: `rows` (16, 32 or 64) jobs of one matrix at a time, from
/// `order` in blocks of `rows` (a block's jobs one matrix's, its unused places [`NONE`]; see [`many_order`]). Each
/// tile's 256 weights are decoded once into the workgroup's memory and thread `(r, c)` sums column `c` for the block's
/// jobs `r`, `r + 16`, .., sixteen products a tile each in `MANY`'s order (so a projection's rows are its own kernel's
/// bit for bit): a workgroup a (tile column, block and split), each job's partial sums to `part[(j * splits + s) *
/// n..]` as [`g_mm_source`]'s. The next tile's words and inputs load while this one's are summed, through two sets of the
/// workgroup's buffers. `p[0]`: n, k, tile words, splits; `p[1]`: words a matrix, the pass's first block.
pub(crate) fn g_many(rows: usize) -> String {
    assert!(matches!(rows, 16 | 32 | 64), "a block of 16, 32 or 64 rows");
    let m = rows / 16;
    let each = |f: &dyn Fn(usize) -> String| (0..m).map(f).collect::<Vec<_>>().join("\n");
    let ids = each(&|i| format!("    let j{i} = ids[r + {}u];", 16 * i));
    let regs = each(&|i| format!("    var xn{i} = 0.0;\n    var acc{i} = 0.0;"));
    let first = each(&|i| format!("        if (j{i} != 0xffffffffu) {{ xn{i} = x[j{i} * k + ks * 16u + c]; }}"));
    let store = each(&|i| format!("        xs[b][(r + {}u) * 16u + c] = xn{i};", 16 * i));
    let next = each(&|i| format!("            if (j{i} != 0xffffffffu) {{ xn{i} = x[j{i} * k + (kt + 1u) * 16u + c]; }}"));
    let sums = each(&|i| {
        format!(
            "            let xb{i} = (r + {}u) * 16u + 4u * q;\n            let x{i} = vec4<f32>(xs[b][xb{i}], xs[b][xb{i} + 1u], xs[b][xb{i} + 2u], xs[b][xb{i} + 3u]);\n            acc{i} = acc{i} + x{i}.x * w4.x;\n            acc{i} = acc{i} + x{i}.y * w4.y;\n            acc{i} = acc{i} + x{i}.z * w4.z;\n            acc{i} = acc{i} + x{i}.w * w4.w;",
            16 * i
        )
    });
    let out = each(&|i| format!("    if (j{i} != 0xffffffffu) {{ part[(j{i} * splits + s) * n + nt * 16u + c] = acc{i}; }}"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> tile: array<array<u32, 64>, 2>;
// a block's inputs, [row][16 of k]; the decoded tile, [column][16 of k] (20 apart, against bank conflicts); each read
// four of k at a time. Scalars, not vec4s: a thread writes one value, and WGSL lets a write to one component of a
// vector in memory write the whole vector (Metal does), so four threads writing one would race
var<workgroup> xs: array<array<f32, {xs_len}>, 2>;
var<workgroup> wt: array<array<f32, 320>, 2>;
var<workgroup> ids: array<u32, {rows}>;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

fn weight(r: u32, c: u32, tw: u32, buf: u32) -> f32 {{
    let nw = tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let jj = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + jj;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    let w0 = start / 32u;
    let sh = 48u - start % 32u;
    let a = tile[buf][w0];
    let b = tile[buf][(w0 + 1u) % nw];
    var code: u32;
    if (sh >= 32u) {{
        code = a >> (sh - 32u);
    }} else {{
        code = (a << (32u - sh)) | (b >> sh);
    }}
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    let blk = p[1].y + wg.z / splits;
    let s = wg.z % splits;
    if (t < {rows}u) {{
        ids[t] = order[blk * {rows}u + t];
    }}
    workgroupBarrier();
    let base = jobs[2u * ids[0]] * p[1].x;
    let r = t / 16u;
    let c = t % 16u;
    let nw = tw / 2u;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    // this thread's rows r + 16 i: it loads their inputs at column c, and sums their outputs at column c
{ids}
{regs}
    var wn = 0u;
    if (ks < ke) {{
        if (t < nw) {{ wn = words[base + (ks * ntiles + nt) * nw + t]; }}
{first}
    }}
    var b = 0u;
    for (var kt = ks; kt < ke; kt = kt + 1u) {{
        if (t < nw) {{ tile[b][t] = wn; }}
{store}
        workgroupBarrier();
        if (kt + 1u < ke) {{
            if (t < nw) {{ wn = words[base + ((kt + 1u) * ntiles + nt) * nw + t]; }}
{next}
        }}
        wt[b][c * 20u + r] = weight(r, c, tw, b);
        workgroupBarrier();
        for (var q = 0u; q < 4u; q = q + 1u) {{
            let wb = c * 20u + 4u * q;
            let w4 = vec4<f32>(wt[b][wb], wt[b][wb + 1u], wt[b][wb + 2u], wt[b][wb + 3u]);
{sums}
        }}
        b = 1u - b;
    }}
{out}
}}
"#,
        xs_len = rows * 16,
    )
}

/// [`g_many`] on the tensor cores (WGSL's cooperative matrices, f16 into f32), for blocks of `rows` (16, 32, 64 or
/// 128) jobs of one matrix: a workgroup 8 tile columns (128 outputs) of a block, `k` a tile row (16) at a time; each
/// step a warp decodes its tile column's 256 codes (a lane its 8, as the one-row kernel's lanes hold them, from the
/// four words they lie in) into the workgroup's memory as f16 (each an f16 exactly) and the block's inputs (f16 too,
/// as the input transform rounds them) beside them, the next step's words and inputs loaded as this one's are
/// multiplied; each warp its 16 outputs by the block's rows. Each job's sums to `part[j * n..]` (one split), through a
/// warp's staging (split `s` of `p[0].w` along k to its own part, `part[(j * splits + s) * n..]`, as [`g_mm_source`]'s).
/// The sums are a matmul's, not the one-row kernel's bit for bit. `p` as for [`g_many`].
pub(crate) fn g_coop(rows: usize) -> String {
    assert!(matches!(rows, 16 | 32 | 64 | 128), "a block of 16, 32, 64 or 128 rows");
    let f = rows / 16;
    let pairs = rows * 8;
    let each = |g: &dyn Fn(usize) -> String| (0..f).map(g).collect::<String>();
    let decl = each(&|i| format!("    var c{i} = coop_mat16x16<f32, C>();\n"));
    let mma = each(&|i| format!("        {{\n            let ib = cur + {}u * S2;\n            let bf = coopLoad<coop_mat16x16<f16, B>>(&xt[ib], s2);\n            c{i} = coopMultiplyAdd(af, bf, c{i});\n        }}\n", i * 16));
    let out = each(&|i| {
        format!(
            "    {{\n        let so = warp * 256u;\n        coopStore(c{i}, &stage[so], 16u);\n        workgroupBarrier();\n        for (var e = l; e < 256u; e += 32u) {{\n            let id = ids[{}u + e / 16u];\n            if (id != 0xffffffffu && live) {{ part[(id * splits + sp) * n + tc * 16u + e % 16u] = stage[so + e]; }}\n        }}\n        workgroupBarrier();\n    }}\n",
            i * 16
        )
    });
    // the inputs a thread loads a step: pairs `t + 256 i` of the block's rows by 16 of k
    let per = pairs.div_ceil(256);
    let xdecl: String = (0..per).map(|i| format!("    var xp{i} = vec2<f32>(0.0);\n")).collect();
    let xload: String = (0..per)
        .map(|i| format!("        {{\n            let q = t + {}u;\n            if (q < {pairs}u) {{\n                let id = ids[q / 8u];\n                xp{i} = vec2<f32>(0.0);\n                if (id != 0xffffffffu) {{ xp{i} = x2[(id * k + kn * 16u) / 2u + q % 8u]; }}\n            }}\n        }}\n", 256 * i))
        .collect();
    let xstore: String = (0..per)
        .map(|i| format!("        {{\n            let q = t + {}u;\n            if (q < {pairs}u) {{ xt[nb + (q / 8u) * S2 + q % 8u] = vec2<f16>(xp{i}); }}\n        }}\n", 256 * i))
        .collect();
    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// a row's stride in the tiles, as f16 pairs: 16 of k and 8 against bank conflicts
const S2: u32 = 12u;
// two steps' weights [output][k] and inputs [row][k]
var<workgroup> wt: array<vec2<f16>, {wt_len}>;
var<workgroup> xt: array<vec2<f16>, {xt_len}>;
var<workgroup> stage: array<f32, 2048>;
// every code's place in a tile, and each lane's first word (the same in every tile)
var<workgroup> places: array<u32, 256>;
// a code's value by its bytes' sum (0 to 1,020), the workgroup's threads four each (as the one-row kernel's)
var<workgroup> values: array<f32, 1024>;
var<workgroup> firsts: array<u32, 32>;
var<workgroup> ids: array<u32, {rows}>;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

// The 16-bit window of code `i` of a tile of `tw` words: (its first word, its shift).
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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let nw = tw / 2u;
    let ntiles = n / 16u;
    // the block, and its split of k (`p[0].w` of them, none empty)
    let splits = p[0].w;
    let blk = p[1].y + wg.z / splits;
    let sp = wg.z % splits;
    let first = window(8u * (t / 8u), tw).x;
    places[t] = place(t, tw, first);
    for (var i = 0u; i < 4u; i = i + 1u) {{
        let sum = 4u * t + i;
        values[sum] = round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
    }}
    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    if (t < {rows}u) {{
        ids[t] = order[blk * {rows}u + t];
    }}
    workgroupBarrier();
    // a block the order left unused (a GPU's grouping sizes the grid for the most blocks it could fill)
    if (workgroupUniformLoad(&ids[0]) == 0xffffffffu) {{
        return;
    }}
    let base = jobs[2u * ids[0]] * p[1].x;
    let warp = t / 32u;
    let l = t % 32u;
    // the warp's tile column (past the matrix: the last, its sums not stored), the lane's codes' places
    let tc = wg.x * 8u + warp;
    let live = tc < ntiles;
    let tcl = min(tc, ntiles - 1u);
    let w0 = firsts[l];
    let a0 = places[8u * l];
    let a1 = places[8u * l + 1u];
    let a2 = places[8u * l + 2u];
    let a3 = places[8u * l + 3u];
    let a4 = places[8u * l + 4u];
    let a5 = places[8u * l + 5u];
    let a6 = places[8u * l + 6u];
    let a7 = places[8u * l + 7u];
    // where its codes go: weights (k 2 (l % 4) + {{0, 1, 8, 9}}, output l / 4 + {{0, 8}}) of the warp's 16
    let wo0 = (warp * 16u + l / 4u) * S2 + l % 4u;
    let wo1 = wo0 + 8u * S2;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = sp * per;
    let ke = min(kts, ks + per);
{decl}{xdecl}    var q0 = 0u;
    var q1 = 0u;
    var q2 = 0u;
    var q3 = 0u;
    // the first step's
    {{
        let kn = ks;
        let wb = base + (kn * ntiles + tcl) * nw;
        q0 = words[wb + w0 % nw];
        q1 = words[wb + (w0 + 1u) % nw];
        q2 = words[wb + (w0 + 2u) % nw];
        q3 = words[wb + (w0 + 3u) % nw];
{xload}        let nb = (ks % 2u) * {half_wt}u;
        wt[nb + wo0] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a0))), f16(decode_at(code_in(q0, q1, q2, q3, a1))));
        wt[nb + wo0 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a2))), f16(decode_at(code_in(q0, q1, q2, q3, a3))));
        wt[nb + wo1] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a4))), f16(decode_at(code_in(q0, q1, q2, q3, a5))));
        wt[nb + wo1 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a6))), f16(decode_at(code_in(q0, q1, q2, q3, a7))));
        {{
            let nb = (ks % 2u) * {half_xt}u;
{xstore}        }}
    }}
    workgroupBarrier();
    for (var kt = ks; kt < ke; kt++) {{
        // the next step (the last's own again, into the buffer no one reads after): loaded before this one's are
        // multiplied, decoded and stored after, with no branch between
        let kn = min(kt + 1u, ke - 1u);
        let wb = base + (kn * ntiles + tcl) * nw;
        q0 = words[wb + w0 % nw];
        q1 = words[wb + (w0 + 1u) % nw];
        q2 = words[wb + (w0 + 2u) % nw];
        q3 = words[wb + (w0 + 3u) % nw];
{xload}        let cur = (kt % 2u) * {half_wt}u;
        let curx = (kt % 2u) * {half_xt}u;
        let s2 = S2;
        let ia = cur + warp * 16u * S2;
        let af = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia], s2);
        {{
            let cur = curx;
{mma}        }}
        let nb = ((kt + 1u) % 2u) * {half_wt}u;
        wt[nb + wo0] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a0))), f16(decode_at(code_in(q0, q1, q2, q3, a1))));
        wt[nb + wo0 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a2))), f16(decode_at(code_in(q0, q1, q2, q3, a3))));
        wt[nb + wo1] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a4))), f16(decode_at(code_in(q0, q1, q2, q3, a5))));
        wt[nb + wo1 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a6))), f16(decode_at(code_in(q0, q1, q2, q3, a7))));
        {{
            let nb = ((kt + 1u) % 2u) * {half_xt}u;
{xstore}        }}
        workgroupBarrier();
    }}
{out}}}
"#,
        wt_len = 2 * 128 * 12,
        xt_len = 2 * rows * 12,
        half_wt = 128 * 12,
        half_xt = rows * 12,
    )
}

/// The pipeline name of [`g_coop`]'s kernel for blocks of `rows`.
pub(crate) fn coop_name(rows: usize) -> &'static str {
    match rows {
        16 => "exl3-coop-16",
        32 => "exl3-coop-32",
        64 => "exl3-coop-64",
        _ => "exl3-coop-128",
    }
}

/// The most rows [`few_kernel`] takes a block.
pub(crate) const FEW_MAX: usize = 8;

/// The matmul for a few rows of one matrix (a check of drafted tokens, a short chunk): [`g_mm_source`]'s lanes, each
/// decoding its eight codes of a tile once and summing them against every row's inputs in the order the one-row
/// kernel sums them (so each row's sums are that kernel's bit for bit). A workgroup a (tile column, block and split);
/// a block is `rows` jobs of one matrix from `order` (its unused places [`NONE`]; a block with none ends at once), each
/// job's partial sums to `part[(j * splits + s) * n..]` as [`g_mm_source`]'s. `p[0]`: n, k, tile words, splits; `p[1]`:
/// words a matrix, the pass's first block.
///
/// Its kernel and the kernel's name for tiles of `tile_words`: written for that rate as the one-row kernel is
/// ([`lanes_fixed`]; the general form under OAIY_EXL3_GENERAL), each made when first met.
pub(crate) fn few_kernel(rows: usize, tile_words: usize) -> (&'static str, &'static str) {
    type Made = [[std::sync::OnceLock<String>; RATES.len() + 1]; FEW_MAX - 1];
    static NAMES: Made = [const { [const { std::sync::OnceLock::new() }; RATES.len() + 1] }; FEW_MAX - 1];
    static SOURCES: Made = [const { [const { std::sync::OnceLock::new() }; RATES.len() + 1] }; FEW_MAX - 1];
    assert!((2..=FEW_MAX).contains(&rows), "a block of 2 to {FEW_MAX} rows");
    let rate = RATES.iter().position(|&rate| rate == tile_words).filter(|_| !general_only());
    let at = rate.unwrap_or(RATES.len());
    let name = NAMES[rows - 2][at].get_or_init(|| match rate {
        Some(_) => format!("exl3-few-{rows}-{tile_words}"),
        None => format!("exl3-few-{rows}"),
    });
    (name, SOURCES[rows - 2][at].get_or_init(|| g_few_source(rows, rate.map(|_| tile_words))))
}

fn g_few_source(rows: usize, fixed: Option<usize>) -> String {
    let Lanes { place, held, loads, codes } = fixed.map_or_else(lanes_general, lanes_fixed);
    let codes: String = codes.iter().enumerate().map(|(jj, code)| format!("        let c{jj} = decode_at({code});\n")).collect();
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let ids = each(&|i| format!("    let j{i} = order[blk * {rows}u + {i}u];"));
    let regs = each(&|i| format!("    var lo{i} = 0.0;\n    var hi{i} = 0.0;"));
    let sums = each(&|i| {
        format!(
            "        if (j{i} != 0xffffffffu) {{
            let xo = (j{i} * k + kt * 16u + rb) / 2u;
            let xa = x2[xo];
            let xb = x2[xo + 4u];
            lo{i} = lo{i} + xa.x * c0;
            lo{i} = lo{i} + xa.y * c1;
            lo{i} = lo{i} + xb.x * c2;
            lo{i} = lo{i} + xb.y * c3;
            hi{i} = hi{i} + xa.x * c4;
            hi{i} = hi{i} + xa.y * c5;
            hi{i} = hi{i} + xb.x * c6;
            hi{i} = hi{i} + xb.y * c7;
        }}"
        )
    });
    let reds = each(&|i| format!("    red[{i}u * 256u + t] = vec2<f32>(lo{i}, hi{i});"));
    let outs = each(&|i| format!("        if (r == {i}u) {{ j = j{i}; }}"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> red: array<vec2<f32>, {red_len}>;
// every code's place in a tile (`place`; in a kernel written for one rate, how far into its word it starts, read of
// a lane's first code alone), the workgroup's threads one each: the same in every tile
var<workgroup> places: array<u32, 256>;
// a code's value by its bytes' sum (0 to 1,020), the workgroup's threads four each (as the one-row kernel's)
var<workgroup> values: array<f32, 1024>;
var<workgroup> firsts: array<u32, 32>;
var<workgroup> lead: u32;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

fn window(i: u32, tw: u32) -> vec2<u32> {{
    let nw = tw / 2u;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec2<u32>(start / 32u, 48u - start % 32u);
}}

fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    let o = 32u * ((wd.x + nw - w0) % nw) + (48u - wd.y);
    return (o >> 4u) | ((16u - (o & 15u)) << 8u);
}}

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

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    let blk = p[1].y + wg.z / splits;
    let s = wg.z % splits;
    if (t == 0u) {{
        lead = order[blk * {rows}u];
    }}
    if (workgroupUniformLoad(&lead) == 0xffffffffu) {{
        return;
    }}
{ids}
    let base = jobs[2u * j0] * p[1].x;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    let warp = t / 32u;
    let l = t % 32u;
    let nw = tw / 2u;
    let rb = 2u * (l % 4u);
    let first = window(8u * (t / 8u), tw).x;
{place}    for (var i = 0u; i < 4u; i = i + 1u) {{
        let sum = 4u * t + i;
        values[sum] = round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
    }}
    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    workgroupBarrier();
    let w0 = firsts[l];
    let o1 = (w0 + 1u) % nw;
    let o2 = (w0 + 2u) % nw;
    let o3 = (w0 + 3u) % nw;
{held}{regs}
    for (var kt = ks + warp; kt < ke; kt = kt + 8u) {{
        let tile = base + (kt * ntiles + nt) * nw;
{loads}{codes}{sums}
    }}
{reds}
    workgroupBarrier();
    // row r's column c: each warp's four lanes of it, the warps in order (as the one-row kernel's `column`)
    if (t < {rows}u * 16u) {{
        let r = t / 16u;
        let c = t % 16u;
        var j = 0xffffffffu;
{outs}
        if (j != 0xffffffffu) {{
            let half = c / 8u;
            let cl = c % 8u;
            var total = 0.0;
            for (var w = 0u; w < 8u; w = w + 1u) {{
                for (var q = 0u; q < 4u; q = q + 1u) {{
                    let v = red[r * 256u + w * 32u + cl * 4u + q];
                    total = total + select(v.x, v.y, half == 1u);
                }}
            }}
            part[(j * splits + s) * n + nt * 16u + c] = total;
        }}
    }}
}}
"#,
        red_len = rows * 256,
    )
}

/// A check's routed experts in blocks for [`few_kernel`], from its down jobs ([`ROUTE_RANK`]'s: pair `j`'s expert, `p[0].x`
/// pairs, at most 256): each expert's pairs (at most `p[0].y`, the block's rows, as a row takes an expert once) a block
/// of their own in their list's order, the blocks in the order their experts first appear; the down jobs' order
/// (`order_d`, `p[0].x` blocks of `p[0].y`) and the gate and up jobs' (`order_gu`: expert block `b`'s gate jobs `2j` in
/// block `2b`, its up jobs `2j + 1` in `2b + 1`), the other places [`NONE`]. One workgroup.
pub(crate) const GROUP: &str = r#"
@group(0) @binding(0) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> order_gu: array<u32>;
@group(0) @binding(7) var<storage, read_write> order_d: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> ex: array<u32, 256>;
// 1 where a pair is its expert's first
var<workgroup> first: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let rows = p[0].y;
    for (var i = t; i < n * rows; i += 256u) {
        order_d[i] = 0xffffffffu;
        order_gu[2u * i] = 0xffffffffu;
        order_gu[2u * i + 1u] = 0xffffffffu;
    }
    if (t < n) {
        ex[t] = jobs[2u * t];
    }
    workgroupBarrier();
    var leader = t;
    var slot = 0u;
    if (t < n) {
        let e = ex[t];
        for (var r = 0u; r < t; r++) {
            if (ex[r] == e) {
                if (leader == t) { leader = r; }
                slot += 1u;
            }
        }
        first[t] = select(0u, 1u, leader == t);
    }
    storageBarrier();
    workgroupBarrier();
    if (t < n) {
        var blk = 0u;
        for (var r = 0u; r < leader; r++) { blk += first[r]; }
        order_d[blk * rows + slot] = t;
        order_gu[2u * blk * rows + slot] = 2u * t;
        order_gu[(2u * blk + 1u) * rows + slot] = 2u * t + 1u;
    }
}
"#;

/// The pipeline name of [`g_many`]'s kernel for blocks of `rows`.
pub(crate) fn many_name(rows: usize) -> &'static str {
    match rows {
        16 => "exl3-many-16",
        32 => "exl3-many-32",
        _ => "exl3-many-64",
    }
}

/// Each job's output transform, a workgroup a (128-block, job): its splits' partial sums added up (in order),
/// rounded to f16, the Hadamard transform of each 128-block, scaled by 1/sqrt(128) and `svh` and rounded (as
/// `Transform::post` before its output map). `p[0]`: n, splits.
const G_POST: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read> svh: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let splits = p[0].y;
    let j = wg.y;
    let m = jobs[2u * j];
    let c = wg.x * 128u + t;
    var v = 0.0;
    for (var s = 0u; s < splits; s++) { v += part[(j * splits + s) * n + c]; }
    sh[t] = half(v);
    workgroupBarrier();
    for (var st = 1u; st < 128u; st *= 2u) {
        if ((t & st) == 0u) {
            let a = sh[t];
            let b = sh[t + st];
            sh[t] = a + b;
            sh[t + st] = a - b;
        }
        workgroupBarrier();
    }
    y[j * n + c] = half(sh[t] * bitcast<f32>(0x3db504f4u) * svh[m * n + c]);
}
"#;

/// Each job's output through its matrix's output map: `y[j, i] = yt[j, omap[m, i]]`. `p[0]`: n.
const G_GATHER: &str = r#"
@group(0) @binding(0) var<storage, read> yt: array<f32>;
@group(0) @binding(1) var<storage, read> omap: array<u32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p[0].x;
    let i = id.x;
    let j = id.y;
    if (i >= n) { return; }
    let m = jobs[2u * j];
    y[j * n + i] = yt[j * n + omap[m * n + i]];
}
"#;

/// OAIY_EXL3_GENERAL: the one-row kernel's general form at every rate (to compare a rate's own with).
fn general_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_EXL3_GENERAL").is_some())
}

/// The chain's one-row kernel for tiles of `tile_words`, and its name: the one written for that rate
/// ([`lanes_fixed`]), each made when a tile of its rate is first met.
pub(crate) fn one_row_kernel(tile_words: usize) -> (&'static str, &'static str) {
    const NAMES: [&str; 11] = ["exl3-mm-16", "exl3-mm-24", "exl3-mm-32", "exl3-mm-40", "exl3-mm-48", "exl3-mm-56", "exl3-mm-64", "exl3-mm-80", "exl3-mm-96", "exl3-mm-112", "exl3-mm-128"];
    static SOURCES: [std::sync::OnceLock<String>; 11] = [const { std::sync::OnceLock::new() }; 11];
    match RATES.iter().position(|&rate| rate == tile_words) {
        Some(at) if !general_only() => (NAMES[at], SOURCES[at].get_or_init(|| g_mm_fixed_source(tile_words))),
        _ => ("exl3-mm", chain_shader("mm")),
    }
}

/// The chain's kernels as WGSL: the input transform (of a SwiGLU's output, "pre-swiglu"), the matmul (a row a job; a
/// prompt's is [`g_many`]), the output transform and its map; each made once.
pub(crate) fn chain_shader(which: &str) -> &'static str {
    static SOURCES: [std::sync::OnceLock<String>; 5] = [const { std::sync::OnceLock::new() }; 5];
    let (at, make): (usize, fn() -> String) = match which {
        "pre" => (0, || format!("{HALF}{G_PRE}")),
        "pre-swiglu" => (1, || format!("{HALF}{G_PRE_SWIGLU}")),
        "mm" => (2, g_mm_source),
        "post" => (3, || format!("{HALF}{G_POST}")),
        _ => (4, || G_GATHER.to_string()),
    };
    SOURCES[at].get_or_init(make)
}
