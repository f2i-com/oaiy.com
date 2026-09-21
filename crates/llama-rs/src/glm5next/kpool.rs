// VENDORED-LOCAL: whole module. GLM-5.3-Flash pooled-indexer KV cache.
//! The `kpool` cache: per-cell indexer state plus the pool geometry the sparse
//! indexer selects over.
//!
//! Reference: llama.cpp `src/llama-kv-cache-kpool.{h,cpp}` and
//! `llm_graph_context::build_inp_kpool` (added by PR #27754, pinned at
//! `86ebfef`), consumed by `glm5next.cpp::build_indexer`.
//!
//! ## What the indexer attends to
//!
//! glm5next's full-attention layers do not attend to all `n_kv` cells. For a
//! query at position `q` the candidate set is two disjoint parts:
//!
//!   * **The tail** — every visible cell from `tail_start(q)` on. These belong
//!     to a pool that is not yet complete, so they have no pooled key and are
//!     always attended densely.
//!   * **Selected pools** — of the complete, fully-visible pools, the indexer
//!     scores one pooled key each and takes the top
//!     [`select_k`] = `top_k / kpool` of them, then expands each to its `kpool`
//!     member cells. With the released `top_k = 2048` and `kpool = 4` that is
//!     512 pools covering 2048 cells.
//!
//! ## Pool geometry
//!
//! A pool is `kpool` consecutive **positions**, anchored on the absolute
//! position rather than on cache order:
//!
//!   * `pool_of(c) = c / kpool`
//!   * `slot_of(c) = c % kpool` — and this is the index into
//!     `indexer_compressor_ape` (`[d_idx, kpool]`), the per-slot positional
//!     encoding added pre-softmax when the pooled key is built.
//!   * A pool is **complete** only when all `kpool` members exist
//!     (`filled == r` in the reference; a straddled pool drops whole).
//!   * A complete pool's pooled key lives in the row of its **last** member,
//!     [`rep_cell`]. The reference is explicit that a partial pool leaves that
//!     slot at 0 and that cell 0 is a real cell, so a pooled read must be
//!     gated on completeness rather than on the slot being non-zero.
//!
//! The reference anchors at the absolute `pos / kpool` ("vLLM, SGLang; not HF's
//! `valid_keys.argmax(-1)`") because that is the only anchor keeping a pool's
//! identity stable from prefill through the decodes that read it.
//!
//! ## Visibility, per query
//!
//! ```text
//! tail_start = (q + 1) / kpool * kpool
//! bo_vis     = tail_start / kpool          // fully-visible complete pools
//! vis(c)     = c <= q
//! pooled(c)  = pool_of(c) < bo_vis
//! sel_mask   = vis && c >= tail_start                 // the dense tail
//! cand_mask  = vis && (pooled || c >= tail_start)     // everything legal
//! pool_bias  = 0 iff complete(p) && p < bo_vis, else -inf
//! ```
//!
//! `sel_mask` is what attention always includes; `cand_mask` bounds what a
//! top-k expansion is allowed to return. `pool_bias` is added to the pool
//! scores before the top-k so an invisible or incomplete pool cannot win a slot.
//!
//! ## Simplifications against the reference
//!
//! `KvCache` here is a contiguous, append-only, single-sequence buffer, so cell
//! `c` always holds position `c`. That removes, as unreachable rather than
//! unimplemented: per-stream partitioning (`strm_of`, `run_off`, `run_len`),
//! the `b_base` rebasing anchor, the `rebuild` flag after a position mutation,
//! the `n_new_max` fixed-width pad with its "repeat a complete pool" filler, and
//! the index clamping the reference needs because `ggml_set_rows` rejects
//! negative indices. Pool members are a contiguous range here
//! ([`pool_members`]) instead of a gathered `pool_cells` table.
//!
//! ## For the CUDA port
//!
//! Fold the pool-score tensor to `[n_pools, n_tps * n_stream]` before the
//! softmax, not `[n_pools, n_tps, n_stream]`. `ggml_soft_max` maps ne2/ne3 onto
//! `gridDim.y`/`z`, which caps at 65535, and at a 1M context `n_pools` is
//! 262,144 — the reference carries the same note, and two open llama.cpp bugs
//! live exactly here (#28729 indexer `soft_max` gridDim.y overflow, #28144
//! SOFT_MAX failure). The host path below is immune; the kernel will not be.
//!
//! **Wired into `forward`: no.** This is the cache half of piece (3); the pooled
//! key's softmax-over-slots with APE belongs to the indexer and is not here —
//! this module owns the "pooled" slot as storage but does not compute it.

use crate::{LlamaError, Result};

/// Heads stored per cell: the indexer key, the compressor gate, and the pool's
/// pooled key. The reference asserts all three are adjacent in a cell.
pub const HEADS_PER_CELL: usize = 3;

/// Index of each head within a cell.
pub const HEAD_KEY: usize = 0;
pub const HEAD_GATE: usize = 1;
pub const HEAD_POOLED: usize = 2;

/// Pools that have at least one member: `ceil(len / kpool)`.
pub const fn n_pools(len: usize, kpool: usize) -> usize {
    if kpool == 0 {
        return 0;
    }
    len.div_ceil(kpool)
}

/// Pools with all `kpool` members present: `len / kpool`. Only these have a
/// pooled key, and only these can be selected.
pub const fn n_complete_pools(len: usize, kpool: usize) -> usize {
    if kpool == 0 {
        return 0;
    }
    len / kpool
}

/// `llama_kpool_select_k`: how many **pools** the top-k picks. A cell-level
/// top-k would be wrong — the reference notes ReLU ties span pool bounds, and a
/// cell cut takes partial pools.
pub fn select_k(n_pools: usize, indexer_top_k: usize, kpool: usize) -> Result<usize> {
    if kpool == 0 || n_pools == 0 {
        return Err(LlamaError::Config(
            "kpool: kpool and n_pools must be > 0".into(),
        ));
    }
    if indexer_top_k % kpool != 0 {
        return Err(LlamaError::Config(format!(
            "kpool: indexer_top_k ({indexer_top_k}) must be a whole number of pools \
             of {kpool}"
        )));
    }
    Ok(n_pools.min(indexer_top_k / kpool))
}

/// The indexer's selection width, `top_k + kpool - 1`.
pub const fn n_select(indexer_top_k: usize, kpool: usize) -> usize {
    indexer_top_k + kpool - 1
}

/// Whether the indexer scores at all, or attention runs dense.
///
/// The reference gates on `cparams.n_ctx > n_select`, deliberately **not** on
/// the live `n_kv`: "gated on n_ctx, not n_kv, which grows and would flip the
/// graph topology mid-run." nrob has no graph-topology constraint, but the
/// comparison target does, so this must gate on the same quantity — the cache
/// capacity — or a prompt straddling `n_select` would diverge from llama.cpp
/// mid-prefill.
pub const fn indexer_scores(max_len: usize, indexer_top_k: usize, kpool: usize) -> bool {
    max_len > n_select(indexer_top_k, kpool)
}

/// Which pool a cell belongs to.
pub const fn pool_of(cell: usize, kpool: usize) -> usize {
    cell / kpool
}

/// A cell's slot within its pool — also its index into
/// `indexer_compressor_ape`.
pub const fn slot_of(cell: usize, kpool: usize) -> usize {
    cell % kpool
}

/// The cell whose row holds a pool's pooled key: its **last** member.
pub const fn rep_cell(pool: usize, kpool: usize) -> usize {
    pool * kpool + kpool - 1
}

/// A pool's member cells. Contiguous here because cell `c` holds position `c`;
/// the reference gathers them through a `pool_cells` table instead.
pub const fn pool_members(pool: usize, kpool: usize) -> std::ops::Range<usize> {
    (pool * kpool)..(pool * kpool + kpool)
}

/// First cell of the dense tail for a query at `q`: `(q + 1) / kpool * kpool`.
pub const fn tail_start(q: usize, kpool: usize) -> usize {
    if kpool == 0 {
        return 0;
    }
    (q + 1) / kpool * kpool
}

/// `bo_vis`: complete pools entirely at or before `q`. The reference tests
/// visibility at a pool's last member, so a pool straddling `q` is not visible
/// at all.
pub const fn n_visible_pools(q: usize, kpool: usize) -> usize {
    if kpool == 0 {
        return 0;
    }
    tail_start(q, kpool) / kpool
}

/// Fill one query's `sel` and `cand` rows, each `len` wide: `0.0` for allowed,
/// `-inf` for masked.
///
/// `sel` is the dense tail attention always includes; `cand` additionally
/// admits cells in complete, visible pools, and bounds what a top-k expansion
/// may return.
pub fn masks_row(
    q: usize,
    len: usize,
    kpool: usize,
    sel: &mut [f32],
    cand: &mut [f32],
) -> Result<()> {
    if kpool == 0 {
        return Err(LlamaError::Config("kpool: kpool must be > 0".into()));
    }
    if sel.len() < len || cand.len() < len {
        return Err(LlamaError::Config(format!(
            "kpool: mask rows are {} / {} wide, need at least len = {len}",
            sel.len(),
            cand.len()
        )));
    }
    let ts = tail_start(q, kpool);
    let bo_vis = n_visible_pools(q, kpool);
    let complete = n_complete_pools(len, kpool);

    for (c, (s, d)) in sel.iter_mut().zip(cand.iter_mut()).enumerate() {
        if c >= len {
            *s = f32::NEG_INFINITY;
            *d = f32::NEG_INFINITY;
            continue;
        }
        let vis = c <= q;
        let tail = c >= ts;
        let p = pool_of(c, kpool);
        // A cell only counts as pooled if its pool is both visible and complete.
        let pooled = p < bo_vis && p < complete;

        *s = if vis && tail { 0.0 } else { f32::NEG_INFINITY };
        *d = if vis && (pooled || tail) {
            0.0
        } else {
            f32::NEG_INFINITY
        };
    }
    Ok(())
}

/// Fill one query's pool-score bias: `0.0` for a complete, visible pool and
/// `-inf` everywhere else, including the slack past `n_pools(len)`.
///
/// `out` is sized on the cache **capacity** so the score row's width is stable
/// across a decode run rather than growing every `kpool` tokens.
pub fn pool_bias_row(q: usize, len: usize, kpool: usize, out: &mut [f32]) -> Result<()> {
    if kpool == 0 {
        return Err(LlamaError::Config("kpool: kpool must be > 0".into()));
    }
    let live = n_pools(len, kpool);
    if out.len() < live {
        return Err(LlamaError::Config(format!(
            "kpool: bias row is {} wide, need at least {live} pools",
            out.len()
        )));
    }
    let bo_vis = n_visible_pools(q, kpool);
    let complete = n_complete_pools(len, kpool);

    for (p, b) in out.iter_mut().enumerate() {
        *b = if p < complete && p < bo_vis {
            0.0
        } else {
            f32::NEG_INFINITY
        };
    }
    Ok(())
}

/// Expand selected pool indices to their member cells, dropping any member past
/// `len`. The reference gathers `pool_cells` rows; the range is contiguous here.
pub fn expand_pools(selected: &[u32], len: usize, kpool: usize, out: &mut Vec<u32>) {
    out.clear();
    if kpool == 0 {
        return;
    }
    for &p in selected {
        for c in pool_members(p as usize, kpool) {
            if c < len {
                out.push(c as u32);
            }
        }
    }
}

/// Per-cell indexer state for the full-attention layers: one `[max_len,
/// HEADS_PER_CELL, d_idx]` buffer per MLA layer.
///
/// The key and gate are written on **every** step, including when the indexer
/// does not score — the reference's `set_input` calls the store unconditionally
/// so that cells written below the `n_select` threshold still have indexer
/// state once a later batch crosses it.
#[derive(Debug)]
pub struct KpoolCache {
    d_idx: usize,
    max_len: usize,
    /// One buffer per MLA layer; empty for layers that have no indexer.
    layers: Vec<Vec<f32>>,
}

impl KpoolCache {
    // VENDORED-LOCAL: GLM-5.3-Flash. For snapshotting a prompt's state to disk.
    /// The per-layer buffers, truncated to what `len` tokens have written.
    ///
    /// The buffers are allocated for `max_len` and filled from the front, so beyond
    /// `len` they are zeros nobody has read yet -- keeping them would make a
    /// snapshot several times bigger for nothing.
    pub fn rows_upto(&self, len: usize) -> Vec<Vec<f32>> {
        let keep = len.min(self.max_len) * self.d_idx;
        self.layers
            .iter()
            .map(|l| l.get(..keep.min(l.len())).unwrap_or(l).to_vec())
            .collect()
    }

    /// Put `rows` back, zeroing whatever lies past them.
    pub fn restore_rows(&mut self, rows: &[Vec<f32>]) -> Result<()> {
        if rows.len() != self.layers.len() {
            return Err(LlamaError::Config(format!(
                "kpool: restoring {} layers into {}",
                rows.len(),
                self.layers.len()
            )));
        }
        for (dst, src) in self.layers.iter_mut().zip(rows) {
            if src.len() > dst.len() {
                return Err(LlamaError::Config(format!(
                    "kpool: restoring {} values into a {}-value layer",
                    src.len(),
                    dst.len()
                )));
            }
            dst[..src.len()].copy_from_slice(src);
            dst[src.len()..].fill(0.0);
        }
        Ok(())
    }

    /// `mla_layers` is the number of full-attention layers, indexed densely:
    /// callers map a block index to its MLA-layer ordinal.
    pub fn new(mla_layers: usize, max_len: usize, d_idx: usize) -> Result<Self> {
        if d_idx == 0 || max_len == 0 {
            return Err(LlamaError::Config(
                "kpool: d_idx and max_len must be > 0".into(),
            ));
        }
        Ok(Self {
            d_idx,
            max_len,
            layers: vec![vec![0.0f32; max_len * HEADS_PER_CELL * d_idx]; mla_layers],
        })
    }

    pub fn d_idx(&self) -> usize {
        self.d_idx
    }
    pub fn max_len(&self) -> usize {
        self.max_len
    }
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.len() * 4).sum()
    }

    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            l.fill(0.0);
        }
    }

    fn offset(&self, cell: usize, head: usize) -> usize {
        (cell * HEADS_PER_CELL + head) * self.d_idx
    }

    fn check(&self, layer: usize, cell: usize) -> Result<()> {
        if layer >= self.layers.len() {
            return Err(LlamaError::Config(format!(
                "kpool: layer {layer} out of range (have {})",
                self.layers.len()
            )));
        }
        if cell >= self.max_len {
            return Err(LlamaError::Config(format!(
                "kpool: cell {cell} past capacity {}",
                self.max_len
            )));
        }
        Ok(())
    }

    /// Write a cell's indexer key and compressor gate. The gate is a **second,
    /// independent** projection of the layer input, not a reuse of the key.
    pub fn store(&mut self, layer: usize, cell: usize, key: &[f32], gate: &[f32]) -> Result<()> {
        self.check(layer, cell)?;
        if key.len() != self.d_idx || gate.len() != self.d_idx {
            return Err(LlamaError::Config(format!(
                "kpool: key/gate are {} / {} wide, expected d_idx = {}",
                key.len(),
                gate.len(),
                self.d_idx
            )));
        }
        let (ko, go) = (self.offset(cell, HEAD_KEY), self.offset(cell, HEAD_GATE));
        let d = self.d_idx;
        self.layers[layer][ko..ko + d].copy_from_slice(key);
        self.layers[layer][go..go + d].copy_from_slice(gate);
        Ok(())
    }

    pub fn key(&self, layer: usize, cell: usize) -> Result<&[f32]> {
        self.check(layer, cell)?;
        let o = self.offset(cell, HEAD_KEY);
        Ok(&self.layers[layer][o..o + self.d_idx])
    }

    pub fn gate(&self, layer: usize, cell: usize) -> Result<&[f32]> {
        self.check(layer, cell)?;
        let o = self.offset(cell, HEAD_GATE);
        Ok(&self.layers[layer][o..o + self.d_idx])
    }

    /// Write a complete pool's pooled key into the row of its last member.
    /// Computing the value is the indexer's job (a softmax over the `kpool`
    /// slots with `indexer_compressor_ape` added pre-softmax); this module only
    /// owns the slot.
    pub fn set_pooled(
        &mut self,
        layer: usize,
        pool: usize,
        kpool: usize,
        pooled: &[f32],
    ) -> Result<()> {
        if kpool == 0 {
            return Err(LlamaError::Config("kpool: kpool must be > 0".into()));
        }
        let cell = rep_cell(pool, kpool);
        self.check(layer, cell)?;
        if pooled.len() != self.d_idx {
            return Err(LlamaError::Config(format!(
                "kpool: pooled key is {} wide, expected d_idx = {}",
                pooled.len(),
                self.d_idx
            )));
        }
        let o = self.offset(cell, HEAD_POOLED);
        let d = self.d_idx;
        self.layers[layer][o..o + d].copy_from_slice(pooled);
        Ok(())
    }

    /// Read a pool's pooled key. The caller must have established that the pool
    /// is complete — an incomplete pool's slot is zeros, and cell 0 is a real
    /// cell, so the slot's contents cannot signal validity.
    pub fn pooled(&self, layer: usize, pool: usize, kpool: usize) -> Result<&[f32]> {
        if kpool == 0 {
            return Err(LlamaError::Config("kpool: kpool must be > 0".into()));
        }
        let cell = rep_cell(pool, kpool);
        self.check(layer, cell)?;
        let o = self.offset(cell, HEAD_POOLED);
        Ok(&self.layers[layer][o..o + self.d_idx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released model's indexer settings.
    const KPOOL: usize = 4;
    const TOP_K: usize = 2048;

    #[test]
    fn released_geometry() {
        let max_len = 1_048_576;
        assert_eq!(n_pools(max_len, KPOOL), 262_144);
        assert_eq!(n_select(TOP_K, KPOOL), 2051);
        assert_eq!(select_k(262_144, TOP_K, KPOOL).unwrap(), 512);
        // 512 pools x 4 cells = the 2048 the indexer is configured for.
        assert_eq!(select_k(262_144, TOP_K, KPOOL).unwrap() * KPOOL, TOP_K);
        assert!(indexer_scores(max_len, TOP_K, KPOOL));
    }

    #[test]
    fn select_k_is_clamped_and_rejects_ragged_top_k() {
        // Fewer pools than top_k/kpool: take them all.
        assert_eq!(select_k(10, TOP_K, KPOOL).unwrap(), 10);
        assert!(select_k(10, 2049, KPOOL).is_err(), "not a whole number of pools");
        assert!(select_k(0, TOP_K, KPOOL).is_err());
        assert!(select_k(10, TOP_K, 0).is_err());
    }

    /// Gated on capacity, like the reference's `n_ctx`, not on live length.
    #[test]
    fn scoring_threshold_is_the_capacity() {
        assert!(!indexer_scores(2051, TOP_K, KPOOL), "exactly n_select is dense");
        assert!(indexer_scores(2052, TOP_K, KPOOL));
        assert!(!indexer_scores(128, TOP_K, KPOOL));
    }

    #[test]
    fn pool_addressing_round_trips() {
        for c in 0..16usize {
            let p = pool_of(c, KPOOL);
            assert_eq!(p, c / 4);
            assert_eq!(slot_of(c, KPOOL), c % 4);
            assert!(pool_members(p, KPOOL).contains(&c));
        }
        // The rep is the LAST member.
        assert_eq!(rep_cell(0, KPOOL), 3);
        assert_eq!(rep_cell(1, KPOOL), 7);
        assert_eq!(pool_members(1, KPOOL), 4..8);
    }

    #[test]
    fn completeness_counts_only_full_pools() {
        assert_eq!((n_pools(8, 4), n_complete_pools(8, 4)), (2, 2));
        assert_eq!((n_pools(9, 4), n_complete_pools(9, 4)), (3, 2));
        assert_eq!((n_pools(11, 4), n_complete_pools(11, 4)), (3, 2));
        assert_eq!((n_pools(0, 4), n_complete_pools(0, 4)), (0, 0));
    }

    /// The three boundary cases that catch an off-by-one in `tail_start` /
    /// `bo_vis`.
    #[test]
    fn visibility_boundaries() {
        // q = 7, len = 8: both pools complete and visible, tail empty.
        assert_eq!(tail_start(7, 4), 8);
        assert_eq!(n_visible_pools(7, 4), 2);
        let (mut s, mut c) = (vec![0.0f32; 8], vec![0.0f32; 8]);
        masks_row(7, 8, 4, &mut s, &mut c).unwrap();
        assert!(s.iter().all(|x| x.is_infinite()), "no tail at a pool boundary");
        assert!(c.iter().all(|x| *x == 0.0), "all 8 cells are pooled candidates");

        // q = 8, len = 9: pools 0-1 visible, tail is exactly {8}.
        assert_eq!(tail_start(8, 4), 8);
        assert_eq!(n_visible_pools(8, 4), 2);
        let (mut s, mut c) = (vec![0.0f32; 9], vec![0.0f32; 9]);
        masks_row(8, 9, 4, &mut s, &mut c).unwrap();
        let tail: Vec<usize> = (0..9).filter(|&j| s[j] == 0.0).collect();
        assert_eq!(tail, vec![8], "the tail is the incomplete pool's cells");
        assert!(c.iter().all(|x| *x == 0.0));
        // Pool 2 is incomplete, so its bias is -inf.
        let mut b = vec![0.0f32; 3];
        pool_bias_row(8, 9, 4, &mut b).unwrap();
        assert_eq!(b[0], 0.0);
        assert_eq!(b[1], 0.0);
        assert!(b[2].is_infinite(), "an incomplete pool cannot be selected");

        // q = 3, len = 8: q sits on a pool boundary, so pool 0 is complete AND
        // fully visible while pool 1 is entirely in the future. There is no
        // tail at all -- every reachable cell is read through a pool.
        assert_eq!(tail_start(3, 4), 4);
        assert_eq!(n_visible_pools(3, 4), 1);
        let mut b = vec![0.0f32; 2];
        pool_bias_row(3, 8, 4, &mut b).unwrap();
        assert_eq!(b[0], 0.0, "pool 0 is complete and ends at q");
        assert!(b[1].is_infinite(), "pool 1 is entirely in the future");

        let (mut s, mut c) = (vec![0.0f32; 8], vec![0.0f32; 8]);
        masks_row(3, 8, 4, &mut s, &mut c).unwrap();
        assert!(
            s.iter().all(|x| x.is_infinite()),
            "a query on a pool boundary has an empty tail"
        );
        for j in 0..=3 {
            assert_eq!(c[j], 0.0, "cell {j} is reachable through pool 0");
        }
        for j in 4..8 {
            assert!(c[j].is_infinite(), "cell {j} is in the future");
        }
    }

    /// No cell may be selectable without also being a candidate, for any
    /// (q, len). This is the invariant the sparse attention relies on.
    #[test]
    fn sel_is_always_a_subset_of_cand() {
        for kpool in 1..=5usize {
            for len in 0..40usize {
                for q in 0..len {
                    let (mut s, mut c) = (vec![0.0f32; len], vec![0.0f32; len]);
                    masks_row(q, len, kpool, &mut s, &mut c).unwrap();
                    for j in 0..len {
                        if s[j] == 0.0 {
                            assert_eq!(c[j], 0.0, "kpool={kpool} len={len} q={q} cell={j}");
                        }
                        // Nothing past the causal boundary is ever allowed.
                        if j > q {
                            assert!(s[j].is_infinite() && c[j].is_infinite());
                        }
                    }
                }
            }
        }
    }

    /// Every cell a pool expansion yields must be an allowed candidate.
    #[test]
    fn expansion_stays_inside_cand_mask() {
        let kpool = 4;
        for len in [8usize, 9, 16, 23, 40] {
            for q in 0..len {
                let live = n_pools(len, kpool);
                let mut bias = vec![0.0f32; live];
                pool_bias_row(q, len, kpool, &mut bias).unwrap();

                // Pools the bias admits are exactly those a top-k could win.
                let admitted: Vec<u32> = (0..live)
                    .filter(|&p| bias[p] == 0.0)
                    .map(|p| p as u32)
                    .collect();

                let mut cells = Vec::new();
                expand_pools(&admitted, len, kpool, &mut cells);

                let (mut s, mut c) = (vec![0.0f32; len], vec![0.0f32; len]);
                masks_row(q, len, kpool, &mut s, &mut c).unwrap();
                for &cell in &cells {
                    assert_eq!(
                        c[cell as usize], 0.0,
                        "len={len} q={q} expanded cell {cell} is not a candidate"
                    );
                }
            }
        }
    }

    #[test]
    fn expansion_is_bounded_by_top_k() {
        let kpool = 4;
        let len = 4096;
        let q = len - 1;
        let live = n_pools(len, kpool);
        let k = select_k(live, TOP_K, kpool).unwrap();

        // Pick the k highest-numbered visible pools, the realistic worst case.
        let bo_vis = n_visible_pools(q, kpool);
        let selected: Vec<u32> = ((bo_vis - k)..bo_vis).map(|p| p as u32).collect();
        let mut cells = Vec::new();
        expand_pools(&selected, len, kpool, &mut cells);

        assert_eq!(cells.len(), TOP_K, "k pools must expand to exactly top_k cells");
        assert!(bo_vis >= TOP_K / kpool);
        let mut sorted = cells.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), cells.len(), "pools must not overlap");
    }

    #[test]
    fn expansion_drops_members_past_len() {
        let mut cells = Vec::new();
        // Pool 2 covers cells 8..12, but len is 10.
        expand_pools(&[2], 10, 4, &mut cells);
        assert_eq!(cells, vec![8, 9]);
    }

    #[test]
    fn cache_stores_key_gate_and_pooled_independently() {
        let d = 4;
        let mut kc = KpoolCache::new(2, 16, d).unwrap();
        assert_eq!(kc.n_layers(), 2);
        assert_eq!(kc.d_idx(), d);
        assert_eq!(kc.size_bytes(), 2 * 16 * HEADS_PER_CELL * d * 4);

        let key = vec![1.0f32, 2.0, 3.0, 4.0];
        let gate = vec![-1.0f32, -2.0, -3.0, -4.0];
        kc.store(1, 5, &key, &gate).unwrap();
        assert_eq!(kc.key(1, 5).unwrap(), &key[..]);
        assert_eq!(kc.gate(1, 5).unwrap(), &gate[..]);
        // A different layer is untouched.
        assert_eq!(kc.key(0, 5).unwrap(), &[0.0; 4][..]);

        // The pooled key for pool 1 lands in cell 7 (its last member).
        let pooled = vec![9.0f32, 8.0, 7.0, 6.0];
        kc.set_pooled(1, 1, 4, &pooled).unwrap();
        assert_eq!(kc.pooled(1, 1, 4).unwrap(), &pooled[..]);
        // ... and does not disturb that cell's key or gate.
        kc.store(1, 7, &key, &gate).unwrap();
        assert_eq!(kc.pooled(1, 1, 4).unwrap(), &pooled[..]);
        assert_eq!(kc.key(1, 7).unwrap(), &key[..]);

        kc.reset();
        assert_eq!(kc.key(1, 5).unwrap(), &[0.0; 4][..]);
    }

    #[test]
    fn cache_rejects_bad_indices_and_widths() {
        let d = 4;
        let mut kc = KpoolCache::new(1, 8, d).unwrap();
        let ok = vec![0.0f32; d];
        assert!(kc.store(1, 0, &ok, &ok).is_err(), "no such layer");
        assert!(kc.store(0, 8, &ok, &ok).is_err(), "past capacity");
        assert!(kc.store(0, 0, &ok[..2], &ok).is_err(), "short key");
        assert!(kc.set_pooled(0, 0, 0, &ok).is_err(), "kpool 0");
        // rep_cell(2, 4) == 11, past the 8-cell capacity.
        assert!(kc.set_pooled(0, 2, 4, &ok).is_err());
        assert!(KpoolCache::new(1, 0, d).is_err());
        assert!(KpoolCache::new(1, 8, 0).is_err());
    }

    #[test]
    fn masks_row_rejects_short_rows() {
        let (mut s, mut c) = (vec![0.0f32; 4], vec![0.0f32; 4]);
        assert!(masks_row(0, 8, 4, &mut s, &mut c).is_err());
        assert!(masks_row(0, 4, 0, &mut s, &mut c).is_err());
        let mut b = vec![0.0f32; 1];
        assert!(pool_bias_row(0, 8, 4, &mut b).is_err());
    }

    /// A row wider than `len` (capacity-sized, as a decode run would use) must
    /// mask the slack rather than leave it readable.
    #[test]
    fn capacity_sized_rows_mask_their_slack() {
        let (mut s, mut c) = (vec![0.0f32; 16], vec![0.0f32; 16]);
        masks_row(8, 9, 4, &mut s, &mut c).unwrap();
        for j in 9..16 {
            assert!(s[j].is_infinite(), "slack cell {j} must be masked");
            assert!(c[j].is_infinite());
        }
        let mut b = vec![0.0f32; 8];
        pool_bias_row(8, 9, 4, &mut b).unwrap();
        for p in 2..8 {
            assert!(b[p].is_infinite(), "slack pool {p} must be masked");
        }
    }
}
