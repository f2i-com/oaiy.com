//! VENDORED-LOCAL: Qwen3-VL's vision tower chained on the backend's device ([`ggml_rs::chain`]), as
//! [`crate::mmproj::Qwen3VlMmProj::forward`] computes it op by op: the patch projection and the learned positions, 27
//! blocks (layer norms, the fused q, k and v, full attention rotated by each patch's row and column, the tanh-GELU
//! MLP), the last norm, each 2 x 2 patches' states side by side through the merger's erf-GELU MLP. Its matrices are
//! f16 on the device (plain floats rounded; a tower stored quantized keeps its EXL3 matrices as they are), a matmul's
//! inputs f32. Op by op the plain matrices are multiplied on the host by a backend that keeps dense weights there:
//! 20 s a picture on any backend, where this is a second on a GPU.
//!
//! The patches are taken in the merger's order (each 2 x 2 block's four together) from the start: attention sees a
//! set, each patch carrying its own position and rotation, so every state is the raster order's, and the merge is
//! then the rows as they lie.

use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, RowNorm, Tensor};

use crate::loader::Weight;
use crate::mmproj::Qwen3VlMmProj;

/// A linear layer on the device: its matrix (f16, or a packed one the device holds) and its bias.
enum Mat {
    Half(DeviceVec),
    Packed(std::sync::Arc<dyn ggml_rs::exl3::PackedLinear>),
}

struct Lin {
    m: Mat,
    b: DeviceVec,
    n: usize,
    k: usize,
}

impl Lin {
    fn of(chain: &dyn DeviceChain, w: &Weight, bias: &Tensor) -> Option<Lin> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        let m = match w {
            Weight::Dense(t) if k % 2 == 0 => Mat::Half(chain.vec_f16_rounded(t.to_host().data())?),
            Weight::Packed(p) if chain.holds_exl3(p.as_ref()) => Mat::Packed(std::sync::Arc::clone(p)),
            _ => return None,
        };
        Some(Lin { m, b: vector(chain, bias.to_host().data()), n, k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows, the inputs as f32.
    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match &self.m {
            Mat::Half(w) => r.matmul_f16_rows_f32(w, self.n, self.k, x, y, rows),
            Mat::Packed(p) => r.exl3_rows(p.as_ref(), x, y, rows),
        }
        r.add_bias_rows(y, &self.b, rows, self.n);
    }
}

fn vector(chain: &dyn DeviceChain, values: &[f32]) -> DeviceVec {
    let v = chain.vec(values.len().max(1));
    chain.upload(&v, values);
    v
}

/// A layer norm's weight less one, then its bias: a modulated norm's scale and shift.
fn norm(chain: &dyn DeviceChain, w: &Tensor, b: &Tensor) -> DeviceVec {
    let mut v: Vec<f32> = w.to_host().data().iter().map(|x| x - 1.0).collect();
    v.extend_from_slice(b.to_host().data());
    vector(chain, &v)
}

struct Block {
    norm1: DeviceVec,
    norm2: DeviceVec,
    qkv: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

/// The tower on the device, made at the first picture.
pub(crate) struct Tower {
    patch: Lin,
    /// The learned positions, `[n, d]`, in the merger's order.
    positions: DeviceVec,
    /// Each patch's rotary pairs (its row's, then its column's), `[n, head_dim]`, in the merger's order.
    table: DeviceVec,
    blocks: Vec<Block>,
    post: DeviceVec,
    mm0: Lin,
    mm2: Lin,
    /// Where the merger's order takes each patch from (its place in the raster).
    order: Vec<usize>,
}

impl Tower {
    /// The tower's weights on `chain`'s device: None where one cannot go there (a matrix of another kind, a value
    /// past f16's range), and the tower runs op by op as before.
    fn of(mm: &Qwen3VlMmProj, chain: &dyn DeviceChain) -> Option<Tower> {
        let cfg = &mm.config;
        let (d, p, hd) = (cfg.embedding_dim, cfg.patch_size, cfg.head_dim);
        let side = cfg.image_size / p;
        if side % 2 != 0 || hd % 4 != 0 || cfg.n_heads * hd != d {
            return None;
        }
        let n = side * side;
        // the merger's order: each 2 x 2 block's (0,0), (0,1), (1,0), (1,1), the blocks row by row
        let order: Vec<usize> = (0..side / 2)
            .flat_map(|by| (0..side / 2).flat_map(move |bx| [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().map(move |(sy, sx)| (2 * by + sy) * side + 2 * bx + sx)))
            .collect();
        let pos_host = mm.position_embd.to_host();
        if pos_host.numel() != n * d {
            return None;
        }
        let positions: Vec<f32> = order.iter().flat_map(|&i| pos_host.data()[i * d..(i + 1) * d].iter().copied()).collect();
        // each patch's rotation, as `multimodal_rope::vision` makes it: a head's first quarter of pairs by its row,
        // the next by its column, pair `j` of each at `10000^(-j / (head_dim / 4))`
        let quarter = hd / 4;
        let table: Vec<f32> = order
            .iter()
            .flat_map(|&i| {
                [(i / side) as f32, (i % side) as f32].into_iter().flat_map(move |at| {
                    (0..quarter).flat_map(move |j| {
                        let (s, c) = (at * 10000f32.powf(-2.0 * j as f32 / (2 * quarter) as f32)).sin_cos();
                        [s, c]
                    })
                })
            })
            .collect();
        let patch_w = mm.patch_embd.to_host();
        let in_dim = 3 * p * p;
        if patch_w.numel() != d * in_dim {
            return None;
        }
        let patch = Lin { m: Mat::Half(chain.vec_f16_rounded(patch_w.data())?), b: vector(chain, mm.patch_embd_b.to_host().data()), n: d, k: in_dim };
        let blocks = mm
            .blocks
            .iter()
            .map(|b| {
                Some(Block {
                    norm1: norm(chain, &b.ln1_w, &b.ln1_b),
                    norm2: norm(chain, &b.ln2_w, &b.ln2_b),
                    qkv: Lin::of(chain, &b.attn_qkv, &b.attn_qkv_b)?,
                    proj: Lin::of(chain, &b.attn_out, &b.attn_out_b)?,
                    fc1: Lin::of(chain, &b.ffn_up, &b.ffn_up_b)?,
                    fc2: Lin::of(chain, &b.ffn_down, &b.ffn_down_b)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Tower {
            patch,
            positions: vector(chain, &positions),
            table: vector(chain, &table),
            blocks,
            post: norm(chain, &mm.post_ln_w, &mm.post_ln_b),
            mm0: Lin::of(chain, &mm.projector.mm0, &mm.projector.mm0_b)?,
            mm2: Lin::of(chain, &mm.projector.mm2, &mm.projector.mm2_b)?,
            order,
        })
    }
}

/// `patches` (`[n, 3 p p]`, in the raster's order, as `unfold_patches_to_host` gives them) through the tower on the
/// backend's device: the soft tokens `[n / 4, width]`, on the host. None where the backend has no chain or the
/// tower's weights cannot go on its device.
pub(crate) fn forward(mm: &Qwen3VlMmProj, patches: &Tensor) -> Option<Tensor> {
    if std::env::var_os("OAIY_NO_CHAIN").is_some() {
        return None;
    }
    let chain = mm.backend.chain()?;
    let tower = mm.chain.get_or_init(|| Tower::of(mm, chain)).as_ref()?;
    let cfg = &mm.config;
    let (d, heads, hd, eps) = (cfg.embedding_dim, cfg.n_heads, cfg.head_dim, cfg.layer_norm_eps);
    let n = tower.order.len();
    let in_dim = tower.patch.k;
    let host = patches.to_host();
    if host.numel() != n * in_dim {
        return None;
    }
    let ordered: Vec<f32> = tower.order.iter().flat_map(|&i| host.data()[i * in_dim..(i + 1) * in_dim].iter().copied()).collect();
    let v = |len: usize| chain.vec(len.max(1));
    let pd = vector(chain, &ordered);
    let (ff, merged, width) = (tower.blocks.first().map_or(0, |b| b.fc1.n), 4 * d, tower.mm2.n);
    let (x, normed, qkv, q, k, vv, kv, o) = (v(n * d), v(n * d), v(n * 3 * d), v(n * d), v(n * d), v(n * d), v(n * 2 * d), v(n * d));
    let att = v(chain.attention_rows_full_out_len(n, heads, hd, n));
    let (f, fg) = (v(n * ff), v(n * ff));
    let (mh, mg, out) = (v(n / 4 * merged), v(n / 4 * merged), v(n / 4 * width));
    let scale = 1.0 / (hd as f32).sqrt();
    {
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        tower.patch.run(r, &pd, &x, n);
        r.add(&x, &tower.positions);
        rec.finish();
    }
    // a recording a block (each one's scratch let go before the next's)
    for b in &tower.blocks {
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        r.norm_mod_rows(&x, &normed, n, d, &b.norm1, 0, Some(d), RowNorm::Layer, eps);
        b.qkv.run(r, &normed, &qkv, n);
        r.copy_cols(&qkv, &q, n, d, 3 * d, 0);
        r.copy_cols(&qkv, &k, n, d, 3 * d, d);
        r.copy_cols(&qkv, &vv, n, d, 3 * d, 2 * d);
        r.rope_rows(&q, n, heads, hd, &tower.table, true);
        r.rope_rows(&k, n, heads, hd, &tower.table, true);
        r.store_rows(&k, &kv, n, d, 0, 2 * d, 0);
        r.store_rows(&vv, &kv, n, d, 0, 2 * d, d);
        r.attention_rows_full(&q, &kv, &att, n, heads, heads, hd, n, scale);
        b.proj.run(r, &att, &o, n);
        r.add(&x, &o);
        r.norm_mod_rows(&x, &normed, n, d, &b.norm2, 0, Some(d), RowNorm::Layer, eps);
        b.fc1.run(r, &normed, &f, n);
        r.gelu(&f, &fg, n * ff);
        b.fc2.run(r, &fg, &o, n);
        r.add(&x, &o);
        rec.finish();
    }
    let mut rec = chain.begin();
    rec.keep_groups(false);
    let r = rec.as_mut();
    // the last norm over each patch's state, then four patches a row as they lie (the merger's order)
    r.norm_mod_rows(&x, &normed, n, d, &tower.post, 0, Some(d), RowNorm::Layer, eps);
    tower.mm0.run(r, &normed, &mh, n / 4);
    r.gelu_erf(&mh, &mg, n / 4 * merged);
    tower.mm2.run(r, &mg, &out, n / 4);
    r.read(&out);
    let values = rec.finish().pop()?;
    Some(Tensor::from_vec(values, vec![n / 4, width]))
}
