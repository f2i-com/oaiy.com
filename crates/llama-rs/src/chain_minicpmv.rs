//! VENDORED-LOCAL: MiniCPM-V 4.7's vision tower chained on the backend's device ([`ggml_rs::chain`]), a tile at a
//! time, as [`crate::mmproj::MiniCpmVMmProj::forward`] computes it op by op: the patch projection and each patch's
//! learned position, the blocks (layer norms, q, k and v, full attention, the tanh-GELU MLP), in 16x the window merger
//! after block `insert` (attention within each 2 x 2 window, on the host: four tokens a window; then each window's four
//! concatenated through its MLP, plus their mean), the last norm, and the final merger (each 2 x 2 of tokens
//! concatenated, normed, its erf-GELU MLP). Op by op every bias and norm went through the host: 8 s a tile on a GPU.
//!
//! The tower has no rotary positions, so attention sees a set: the patches are taken in the mergers' order from the
//! start (each final token's 2 x 2 of window merges, each of those a window's 2 x 2 of patches), and every merge is then
//! rows as they lie, the tokens coming out row by row.

use ggml_rs::{DeviceChain, DeviceVec, RowNorm, Tensor};

use crate::chain_mmproj::{norm, vector, Lin, Mat};
use crate::mmproj::MiniCpmVMmProj;

struct Block {
    norm1: DeviceVec,
    norm2: DeviceVec,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    fc1: Lin,
    fc2: Lin,
}

struct Window {
    insert: usize,
    norm1: DeviceVec,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ds_norm: DeviceVec,
    up: Lin,
    down: Lin,
    /// `[d, 4 d]`: a window's four rows' mean, as a matmul.
    mean: Lin,
}

/// The tower on the device, made at the first tile.
pub(crate) struct Tower {
    patch: Lin,
    blocks: Vec<Block>,
    window: Option<Window>,
    post: DeviceVec,
    merger_norm: DeviceVec,
    up: Lin,
    down: Lin,
}

/// [`MiniCpmVMmProj`]'s tower on its device once a tile has asked for it (None inside where it cannot go there), and
/// whether the host's copy of its matrices was let go ([`MiniCpmVMmProj::keep_on_device`]).
#[derive(Default)]
pub struct TowerCell(std::sync::OnceLock<Option<Tower>>, pub(crate) std::sync::atomic::AtomicBool);

/// VENDORED-LOCAL: `mm`'s tower made on its backend's device now (where a first tile would make it): whether it is there.
pub(crate) fn prepare(mm: &MiniCpmVMmProj) -> bool {
    if std::env::var_os("OAIY_NO_CHAIN").is_some() {
        return false;
    }
    let Some(chain) = mm.backend.chain() else { return false };
    mm.chain.0.get_or_init(|| Tower::of(mm, chain)).is_some()
}

impl std::fmt::Debug for TowerCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0.get() {
            None => "TowerCell(not yet)",
            Some(None) => "TowerCell(none)",
            Some(Some(_)) => "TowerCell(on the device)",
        })
    }
}

impl Tower {
    fn of(mm: &MiniCpmVMmProj, chain: &dyn DeviceChain) -> Option<Tower> {
        let d = mm.config.embedding_dim;
        let p = mm.config.patch_size;
        let in_dim = 3 * p * p;
        let patch_w = mm.patch_embd.to_host();
        if patch_w.numel() != d * in_dim || mm.config.n_heads * mm.config.head_dim != d {
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
                    q: Lin::of(chain, &b.attn_q, &b.attn_q_b)?,
                    k: Lin::of(chain, &b.attn_k, &b.attn_k_b)?,
                    v: Lin::of(chain, &b.attn_v, &b.attn_v_b)?,
                    o: Lin::of(chain, &b.attn_out, &b.attn_out_b)?,
                    fc1: Lin::of(chain, &b.ffn_up, &b.ffn_up_b)?,
                    fc2: Lin::of(chain, &b.ffn_down, &b.ffn_down_b)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let window = match &mm.window {
            None => None,
            Some(w) => {
                let mut mean = vec![0f32; d * 4 * d];
                for j in 0..d {
                    for q in 0..4 {
                        mean[j * 4 * d + q * d + j] = 0.25;
                    }
                }
                Some(Window {
                    insert: w.insert,
                    norm1: norm(chain, &w.ln1.0, &w.ln1.1),
                    q: Lin::of(chain, &w.q, &w.q_b)?,
                    k: Lin::of(chain, &w.k, &w.k_b)?,
                    v: Lin::of(chain, &w.v, &w.v_b)?,
                    o: Lin::of(chain, &w.o, &w.o_b)?,
                    ds_norm: norm(chain, &w.ds_ln.0, &w.ds_ln.1),
                    up: Lin::of(chain, &w.up, &w.up_b)?,
                    down: Lin::of(chain, &w.down, &w.down_b)?,
                    mean: Lin { m: Mat::Half(chain.vec_f16_rounded(&mean)?), b: vector(chain, &vec![0.; d]), n: d, k: 4 * d },
                })
            }
        };
        Some(Tower {
            patch,
            blocks,
            window,
            post: norm(chain, &mm.post_ln.0, &mm.post_ln.1),
            merger_norm: norm(chain, &mm.merger_ln.0, &mm.merger_ln.1),
            up: Lin::of(chain, &mm.merger_up, &mm.merger_up_b)?,
            down: Lin::of(chain, &mm.merger_down, &mm.merger_down_b)?,
        })
    }
}

/// The patches of a `gh` x `gw` grid in the mergers' order, each its place in the raster: with the window merger each
/// output token's 2 x 2 of windows (row by row over the tokens), each window's 2 x 2 of patches; without it each output
/// token's 2 x 2 of patches. A quad is top left, top right, bottom left, bottom right.
fn merge_order(gh: usize, gw: usize, window: bool) -> Vec<usize> {
    const QUAD: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];
    let mut order = Vec::with_capacity(gh * gw);
    if window {
        for i in 0..gh / 4 {
            for j in 0..gw / 4 {
                for (a, b) in QUAD {
                    for (dy, dx) in QUAD {
                        order.push((4 * i + 2 * a + dy) * gw + 4 * j + 2 * b + dx);
                    }
                }
            }
        }
    } else {
        for i in 0..gh / 2 {
            for j in 0..gw / 2 {
                for (dy, dx) in QUAD {
                    order.push((2 * i + dy) * gw + 2 * j + dx);
                }
            }
        }
    }
    order
}

/// A tile (`[3, H, W]`) through the tower on the backend's device: its tokens `[H W / (14 merge)^2, width]`, row by
/// row, on the host. None where the backend has no chain or the tower's weights cannot go on its device.
pub(crate) fn forward(mm: &MiniCpmVMmProj, cell: &TowerCell, tile: &Tensor) -> Option<Tensor> {
    if std::env::var_os("OAIY_NO_CHAIN").is_some() {
        return None;
    }
    let chain = mm.backend.chain()?;
    let tower = cell.0.get_or_init(|| Tower::of(mm, chain)).as_ref()?;
    let cfg = &mm.config;
    let (d, heads, hd, eps, p) = (cfg.embedding_dim, cfg.n_heads, cfg.head_dim, cfg.layer_norm_eps, cfg.patch_size);
    let shape = tile.shape();
    let (gh, gw) = (shape[1] / p, shape[2] / p);
    let order = merge_order(gh, gw, tower.window.is_some());
    let n = order.len();
    let in_dim = tower.patch.k;
    let patches = crate::mmproj::unfold_patches_to_host(tile, p).ok()?;
    let ordered: Vec<f32> = order.iter().flat_map(|&i| patches.data()[i * in_dim..(i + 1) * in_dim].iter().copied()).collect();
    // each patch's learned position: the 70 x 70 grid's nearest
    let table = mm.position_embd.to_host();
    let positions: Vec<f32> = order.iter().flat_map(|&i| { let id = (70 * (i / gw) / gh) * 70 + 70 * (i % gw) / gw; table.data()[id * d..(id + 1) * d].iter().copied() }).collect();
    let v = |len: usize| chain.vec(len.max(1));
    let ff = tower.blocks.first().map_or(0, |b| b.fc1.n);
    let width = tower.down.n;
    let (x, normed, q, k, vv, kv, o) = (v(n * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * 2 * d), v(n * d));
    let att = v(chain.attention_rows_full_out_len(n, heads, hd, n));
    let (f, fg) = (v(n * ff), v(n * ff));
    let scale = 1.0 / (hd as f32).sqrt();
    let pd = vector(chain, &ordered);
    let pos = vector(chain, &positions);
    {
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        tower.patch.run(r, &pd, &x, n);
        r.add(&x, &pos);
        rec.finish();
    }
    let mut rows = n;
    for (i, b) in tower.blocks.iter().enumerate() {
        let mut rec = chain.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        r.norm_mod_rows(&x, &normed, rows, d, &b.norm1, 0, Some(d), RowNorm::Layer, eps);
        b.q.run(r, &normed, &q, rows);
        b.k.run(r, &normed, &k, rows);
        b.v.run(r, &normed, &vv, rows);
        r.store_rows(&k, &kv, rows, d, 0, 2 * d, 0);
        r.store_rows(&vv, &kv, rows, d, 0, 2 * d, d);
        r.attention_rows_full(&q, &kv, &att, rows, heads, heads, hd, rows, scale);
        b.o.run(r, &att, &o, rows);
        r.add(&x, &o);
        r.norm_mod_rows(&x, &normed, rows, d, &b.norm2, 0, Some(d), RowNorm::Layer, eps);
        b.fc1.run(r, &normed, &f, rows);
        r.gelu(&f, &fg, rows * ff);
        b.fc2.run(r, &fg, &o, rows);
        r.add(&x, &o);
        rec.finish();
        if let Some(w) = tower.window.as_ref().filter(|w| w.insert == i) {
            // attention within each window (four rows as they lie), on the host
            let mut rec = chain.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            r.norm_mod_rows(&x, &normed, rows, d, &w.norm1, 0, Some(d), RowNorm::Layer, eps);
            w.q.run(r, &normed, &q, rows);
            w.k.run(r, &normed, &k, rows);
            w.v.run(r, &normed, &vv, rows);
            r.read(&q);
            r.read(&k);
            r.read(&vv);
            let mut read = rec.finish();
            let (vh, kh, qh) = (read.pop()?, read.pop()?, read.pop()?);
            let mut out = vec![0f32; rows * d];
            for win in 0..rows / 4 {
                for h in 0..heads {
                    let at = |r: usize| r * d + h * hd;
                    for a in 0..4 {
                        let ra = 4 * win + a;
                        let mut s = [0f32; 4];
                        for (bb, sv) in s.iter_mut().enumerate() {
                            let rb = 4 * win + bb;
                            let (qa, kb) = (at(ra), at(rb));
                            *sv = (0..hd).map(|e| qh[qa + e] * kh[kb + e]).sum::<f32>() * scale;
                        }
                        let most = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = s.iter().map(|sv| (sv - most).exp()).collect();
                        let sum: f32 = e.iter().sum();
                        let dst = at(ra);
                        for (bb, pe) in e.iter().enumerate() {
                            let src = at(4 * win + bb);
                            for c in 0..hd {
                                out[dst + c] += pe / sum * vh[src + c];
                            }
                        }
                    }
                }
            }
            let host_att = vector(chain, &out);
            // then each window's four to one: their concatenation's MLP plus their mean
            let mut rec = chain.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            w.o.run(r, &host_att, &o, rows);
            r.add(&x, &o);
            let quads = rows / 4;
            r.norm_mod_rows(&x, &normed, quads, 4 * d, &w.ds_norm, 0, Some(4 * d), RowNorm::Layer, eps);
            w.up.run(r, &normed, &f, quads);
            r.gelu(&f, &fg, quads * w.up.n);
            w.down.run(r, &fg, &o, quads);
            w.mean.run(r, &x, &q, quads);
            r.add(&o, &q);
            r.copy(&o, 0, &x, 0, quads * d);
            rec.finish();
            rows = quads;
        }
    }
    let mut rec = chain.begin();
    rec.keep_groups(false);
    let r = rec.as_mut();
    let quads = rows / 4;
    r.norm_mod_rows(&x, &normed, rows, d, &tower.post, 0, Some(d), RowNorm::Layer, eps);
    // the final merger: four rows as they lie
    let mh = v(quads * tower.up.n);
    let mg = v(quads * tower.up.n);
    let out = v(quads * width);
    r.norm_mod_rows(&normed, &x, quads, 4 * d, &tower.merger_norm, 0, Some(4 * d), RowNorm::Layer, eps);
    tower.up.run(r, &x, &mh, quads);
    r.gelu_erf(&mh, &mg, quads * tower.up.n);
    tower.down.run(r, &mg, &out, quads);
    r.read(&out);
    let values = rec.finish().pop()?;
    Some(Tensor::from_vec(values, vec![quads, width]))
}
