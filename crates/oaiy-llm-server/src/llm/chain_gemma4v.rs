//! Gemma 4's vision tower chained on the backend's device ([`ggml_rs::chain`]), as Transformers'
//! `Gemma4VisionModel` and llama.cpp's `gemma4v` graph compute it: the patches (pixels to `[-1, 1]`) through the patch
//! projection plus each patch's column's and row's learned positions; 16 blocks of RMS norms on both sides of the
//! attention and the MLP, q and k normed per head and v without a weight, a head's first half rotated by its patch's
//! column and its second by its row (NeoX pairs, base 100), full attention at scale 1, the gated tanh-GELU MLP, each
//! linear layer's input and output clamped to its learned range; then each 3 x 3 patches' mean times `sqrt(d)`, an RMS
//! norm without a weight, and the projection into the language model's width. Its matrices are f16 on the device (the
//! checkpoint's own).
//!
//! The patches are taken in the pooling's order (each 3 x 3 block's nine together, the blocks row by row): attention
//! sees a set, each patch carrying its own position, so every state is the raster order's, and the pooling is then a
//! matmul of nine rows at a time.

use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, Tensor};

use crate::llm::loader::Weight;
use crate::llm::mmproj::{Gemma4VBlock, Gemma4VMmProj};

/// The pooling's kernel (Gemma 4's `pooling_kernel_size`).
pub const POOL: usize = 3;

fn vector(chain: &dyn DeviceChain, values: &[f32]) -> DeviceVec {
    let v = chain.vec(values.len().max(1));
    chain.upload(&v, values);
    v
}

fn host(t: &Tensor) -> Vec<f32> {
    if t.is_device() { t.to_host().data().to_vec() } else { t.data().to_vec() }
}

/// A weight's values as f32 (`[n, k]`).
fn dense(w: &Weight) -> Option<(Vec<f32>, usize, usize)> {
    match w {
        Weight::Dense(t) if t.rank() == 2 => Some((host(t), t.dim(0), t.dim(1))),
        _ => None,
    }
}

/// A clamp's range, None where it lets everything through.
fn range(lo: f32, hi: f32) -> Option<(f32, f32)> {
    (lo > f32::MIN || hi < f32::MAX).then_some((lo, hi))
}

/// A linear layer on the device: its f16 matrix (`[n, k]`), its output's clamp.
struct Lin {
    w: DeviceVec,
    n: usize,
    k: usize,
    out: Option<(f32, f32)>,
}

impl Lin {
    /// `ws` side by side (their rows one after another, each `k` wide), clamped to `out`.
    fn of(chain: &dyn DeviceChain, ws: &[&Weight], out: Option<(f32, f32)>) -> Option<Lin> {
        let mut values = Vec::new();
        let (mut n, mut k) = (0, None);
        for w in ws {
            let (v, wn, wk) = dense(w)?;
            if k.is_some_and(|k| k != wk) {
                return None;
            }
            k = Some(wk);
            n += wn;
            values.extend_from_slice(&v);
        }
        let k = k?;
        if k % 2 != 0 {
            return None;
        }
        Some(Lin { w: chain.vec_f16(&values).or_else(|| chain.vec_f16_rounded(&values))?, n, k, out })
    }

    /// `y[r] = clamp(W x[r])` for `rows` rows.
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows_ranged(&self.w, self.n, self.k, x, y, rows);
        if let Some((lo, hi)) = self.out {
            r.clamp_in_place(y, rows * self.n, lo, hi);
        }
    }
}

struct Block {
    ln1: DeviceVec,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    post_attn: DeviceVec,
    ln2: DeviceVec,
    post_ffn: DeviceVec,
    /// q, k and v's shared input clamp; q, k and v, each its own output clamp.
    qkv_in: Option<(f32, f32)>,
    q: Lin,
    k: Lin,
    v: Lin,
    o_in: Option<(f32, f32)>,
    o: Lin,
    /// The gate and up side by side (their shared clamps), and down.
    gu_in: Option<(f32, f32)>,
    gate_up: Lin,
    down_in: Option<(f32, f32)>,
    down: Lin,
}

impl Block {
    fn of(chain: &dyn DeviceChain, b: &Gemma4VBlock) -> Option<Block> {
        let c = &b.clamps;
        // q, k and v read one normed input: one clamp of it, where theirs agree (Gemma 4's do); so the gate and up
        if c.attn_q.input != c.attn_k.input || c.attn_q.input != c.attn_v.input || c.ffn_gate.input != c.ffn_up.input || c.ffn_gate.output != c.ffn_up.output {
            return None;
        }
        let r = |(lo, hi): (f32, f32)| range(lo, hi);
        Some(Block {
            ln1: vector(chain, &host(&b.ln1)),
            q_norm: vector(chain, &host(&b.attn_q_norm)),
            k_norm: vector(chain, &host(&b.attn_k_norm)),
            post_attn: vector(chain, &host(&b.attn_post)),
            ln2: vector(chain, &host(&b.ln2)),
            post_ffn: vector(chain, &host(&b.ffn_post)),
            qkv_in: r(c.attn_q.input),
            q: Lin::of(chain, &[&b.attn_q], r(c.attn_q.output))?,
            k: Lin::of(chain, &[&b.attn_k], r(c.attn_k.output))?,
            v: Lin::of(chain, &[&b.attn_v], r(c.attn_v.output))?,
            o_in: r(c.attn_out.input),
            o: Lin::of(chain, &[&b.attn_out], r(c.attn_out.output))?,
            gu_in: r(c.ffn_gate.input),
            gate_up: Lin::of(chain, &[&b.ffn_gate, &b.ffn_up], r(c.ffn_gate.output))?,
            down_in: r(c.ffn_down.input),
            down: Lin::of(chain, &[&b.ffn_down], r(c.ffn_down.output))?,
        })
    }
}

/// The tower on the device, made at the first picture.
pub(crate) struct Tower {
    patch: Lin,
    blocks: Vec<Block>,
    /// Each 3 x 3 patches' mean times `sqrt(d)`: `[d, 9 d]` (f32), nine rows of `d` side by side into one.
    pool: DeviceVec,
    ones: DeviceVec,
    proj: Lin,
}

impl Tower {
    pub(crate) fn of(mm: &Gemma4VMmProj, chain: &dyn DeviceChain) -> Option<Tower> {
        let cfg = &mm.config;
        let (d, p) = (cfg.embedding_dim, cfg.patch_size);
        if cfg.n_heads * cfg.head_dim != d || cfg.head_dim % 4 != 0 {
            return None;
        }
        let patch_w = host(&mm.patch_embd);
        if patch_w.len() != d * 3 * p * p {
            return None;
        }
        let patch = Lin { w: chain.vec_f16(&patch_w).or_else(|| chain.vec_f16_rounded(&patch_w))?, n: d, k: 3 * p * p, out: None };
        let blocks = mm.blocks.iter().map(|b| Block::of(chain, b)).collect::<Option<Vec<_>>>()?;
        let k = POOL * POOL;
        let s = (d as f32).sqrt() / k as f32;
        let mut pool = vec![0f32; d * k * d];
        for o in 0..d {
            for j in 0..k {
                pool[o * k * d + j * d + o] = s;
            }
        }
        let c = &mm.projector.clamps;
        Some(Tower {
            patch,
            blocks,
            pool: vector(chain, &pool),
            ones: vector(chain, &vec![1.0; d]),
            proj: Lin::of(chain, &[&mm.projector.input_projection], range(c.output.0, c.output.1))?,
        })
    }
}

/// A picture's patches in the pooling's order: `(column, row)` of each, nine at a time (each 3 x 3 block's, the blocks
/// row by row), of a grid `cols` by `rows` patches.
pub fn pooled_order(cols: usize, rows: usize) -> Vec<(usize, usize)> {
    (0..rows / POOL)
        .flat_map(|by| (0..cols / POOL).flat_map(move |bx| (0..POOL).flat_map(move |sy| (0..POOL).map(move |sx| (bx * POOL + sx, by * POOL + sy)))))
        .collect()
}

/// `image` (`[3, h, w]`, values in `[0, 1]`, each side a multiple of 48) through the tower on the backend's device:
/// the soft tokens `[(h / 48) (w / 48), width]`, row by row, on the host. None where the backend has no chain or the
/// tower's weights cannot go on its device.
pub(crate) fn forward(mm: &Gemma4VMmProj, image: &Tensor) -> Option<Tensor> {
    if std::env::var_os("OAIY_NO_CHAIN").is_some() {
        return None;
    }
    let chain = mm.backend.chain()?;
    let tower = mm.chain.get_or_init(|| Tower::of(mm, chain)).as_ref()?;
    let cfg = &mm.config;
    let (d, heads, hd, p, eps) = (cfg.embedding_dim, cfg.n_heads, cfg.head_dim, cfg.patch_size, cfg.layer_norm_eps);
    let img = image.to_host();
    let (h, w) = (img.dim(1), img.dim(2));
    let (rows, cols) = (h / p, w / p);
    if img.dim(0) != 3 || rows % POOL != 0 || cols % POOL != 0 || rows == 0 || cols == 0 {
        return None;
    }
    let order = pooled_order(cols, rows);
    let n = order.len();
    let pix = img.data();
    // each patch's pixels as the projection takes them (channel, then row, then column), from [0, 1] to [-1, 1]
    let mut patches = Vec::with_capacity(n * 3 * p * p);
    for &(x, y) in &order {
        for c in 0..3 {
            for py in 0..p {
                let at = c * h * w + (y * p + py) * w + x * p;
                patches.extend(pix[at..at + p].iter().map(|v| 2.0 * v - 1.0));
            }
        }
    }
    // the learned positions: the column's (table 0) plus the row's (table 1)
    let pe = host(&mm.position_embd);
    let max_pos = mm.position_embd.dim(1);
    if cols > max_pos || rows > max_pos {
        return None;
    }
    let mut positions = Vec::with_capacity(n * d);
    for &(x, y) in &order {
        positions.extend((0..d).map(|j| pe[x * d + j] + pe[(max_pos + y) * d + j]));
    }
    // each half head's rotation: the first by the patch's column, the second by its row, pair `j` at `100^(-j / (hd / 4))`
    let quarter = hd / 4;
    let table: Vec<f32> = order
        .iter()
        .flat_map(|&(x, y)| {
            (0..heads).flat_map(move |_| {
                [x as f32, y as f32].into_iter().flat_map(move |at| {
                    (0..quarter).flat_map(move |j| {
                        let (s, c) = (at * 100f32.powf(-2.0 * j as f32 / (2 * quarter) as f32)).sin_cos();
                        [s, c]
                    })
                })
            })
        })
        .collect();
    let v = |len: usize| chain.vec(len.max(1));
    let ff = tower.blocks.first().map_or(0, |b| b.down.k);
    let (x, xn, q, k, vv, qn, kn, vn, kv, o, on) =
        (v(n * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * 2 * d), v(n * d), v(n * d));
    let att = v(chain.attention_rows_full_out_len(n, heads, hd, n));
    let (gu, act) = (v(n * 2 * ff), v(n * ff));
    let (pooled, pn, out) = (v(n / 9 * d), v(n / 9 * d), v(n / 9 * tower.proj.n));
    {
        let pd = vector(chain, &patches);
        let pos = vector(chain, &positions);
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        tower.patch.run(r, &pd, &x, n);
        r.add(&x, &pos);
        rec.finish();
    }
    let table = vector(chain, &table);
    // a recording a block (each one's scratch let go before the next's)
    for b in &tower.blocks {
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        r.rmsnorm_rows(&x, &b.ln1, &xn, n, eps);
        if let Some((lo, hi)) = b.qkv_in {
            r.clamp_in_place(&xn, n * d, lo, hi);
        }
        b.q.run(r, &xn, &q, n);
        b.k.run(r, &xn, &k, n);
        b.v.run(r, &xn, &vv, n);
        r.rmsnorm_rows(&q, &b.q_norm, &qn, n * heads, eps);
        r.rmsnorm_rows(&k, &b.k_norm, &kn, n * heads, eps);
        r.rmsnorm_rows(&vv, &tower.ones, &vn, n * heads, eps);
        // each head's halves as heads of their own, each its own row of the table
        r.rope_rows(&qn, n * heads * 2, 1, hd / 2, &table, true);
        r.rope_rows(&kn, n * heads * 2, 1, hd / 2, &table, true);
        r.store_rows(&kn, &kv, n, d, 0, 2 * d, 0);
        r.store_rows(&vn, &kv, n, d, 0, 2 * d, d);
        r.attention_rows_full(&qn, &kv, &att, n, heads, heads, hd, n, 1.0);
        if let Some((lo, hi)) = b.o_in {
            r.clamp_in_place(&att, n * d, lo, hi);
        }
        b.o.run(r, &att, &o, n);
        r.rmsnorm_rows(&o, &b.post_attn, &on, n, eps);
        r.add(&x, &on);
        r.rmsnorm_rows(&x, &b.ln2, &xn, n, eps);
        if let Some((lo, hi)) = b.gu_in {
            r.clamp_in_place(&xn, n * d, lo, hi);
        }
        // the gate's outputs then the up's, a row each, as the fused GELU takes them
        b.gate_up.run(r, &xn, &gu, n);
        r.gelu_mul_split_rows(&gu, &act, n);
        if let Some((lo, hi)) = b.down_in {
            r.clamp_in_place(&act, n * ff, lo, hi);
        }
        b.down.run(r, &act, &o, n);
        r.rmsnorm_rows(&o, &b.post_ffn, &on, n, eps);
        r.add(&x, &on);
        rec.finish();
    }
    let mut rec = chain.begin();
    rec.keep_groups(false);
    let r = rec.as_mut();
    // each block's nine patches' mean (times sqrt(d)), an RMS norm without a weight, the projection
    r.matmul_f32_rows(&tower.pool, d, POOL * POOL * d, &x, &pooled, n / 9);
    r.rmsnorm_rows(&pooled, &tower.ones, &pn, n / 9, eps);
    tower.proj.run(r, &pn, &out, n / 9);
    r.read(&out);
    let values = rec.finish().pop()?;
    Some(Tensor::from_vec(values, vec![n / 9, tower.proj.n]))
}

