//! MiniCPM-V 4.7's vision projector (`clip.projector_type = "minicpmv4_7"`), as transformers'
//! `MiniCPMV4_7ForConditionalGeneration` and its processor make a picture's tokens: the picture cut into an overview
//! and up to nine slices ([`plan`]), each tile through a SigLIP tower with a window-attention merger after its
//! seventh block (16x: four patches a token there, then the final merger's four again; 4x: the final merger's alone),
//! and every token of the prompt given its place on the picture's canvas ([`canvas_positions`]: the time channel held
//! at the canvas's origin, rows and columns counted from it).

use super::*;

/// The processor's defaults (`MiniCPMV4_6ImageProcessor`): a tile about this many pixels a side, of 14-pixel patches,
/// each side a multiple of four patches (the two 2x2 merges).
pub const SCALE_RESOLUTION: usize = 448;
pub const PATCH: usize = 14;
pub const MAX_SLICES: usize = 9;

/// Python's `round`: halves to the even neighbour.
fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r }
}

/// `ensure_divide`: `length` rounded to a multiple of `divisor`, at least one.
fn ensure_divide(length: f64, divisor: usize) -> usize {
    ((round_even(length / divisor as f64) as usize) * divisor).max(divisor)
}

/// `find_best_resize`: a tile's size (height, width) for a picture of `(h, w)` pixels, scaled to about
/// `scale` x `scale` where it is larger (or always, `upscale`), each side a multiple of four patches.
fn best_resize(h: f64, w: f64, scale: usize, upscale: bool) -> (usize, usize) {
    let (mut height, mut width) = (h, w);
    if h * w > (scale * scale) as f64 || upscale {
        let ratio = w / h;
        height = (scale as f64 / ratio.sqrt()).trunc();
        width = (height * ratio).trunc();
    }
    (ensure_divide(height, PATCH * 4), ensure_divide(width, PATCH * 4))
}

/// `get_sliced_grid`: the (rows, columns) a picture is cut into, None where it is small enough whole: the count of
/// tiles its area asks for (at most `max_slices`), give or take one, its shape the nearest the picture's.
fn sliced_grid(h: usize, w: usize, max_slices: usize, scale: usize) -> Option<(usize, usize)> {
    let log_ratio = (w as f64 / h as f64).ln();
    let ratio = (w * h) as f64 / (scale * scale) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slices);
    if multiple <= 1 {
        return None;
    }
    let (mut best, mut least) = ((1, 1), f64::INFINITY);
    for n in [multiple - 1, multiple, multiple + 1] {
        if n == 1 || n > max_slices {
            continue;
        }
        for rows in 1..=n {
            if n % rows == 0 {
                let cols = n / rows;
                let error = (log_ratio - (cols as f64 / rows as f64).ln()).abs();
                if error < least {
                    (best, least) = ((rows, cols), error);
                } else if error == least && rows > best.0 {
                    best = (rows, cols);
                }
            }
        }
    }
    Some(best)
}

/// How a picture of `h` x `w` pixels becomes tiles: its overview's size, and where it is sliced, the grid and the
/// size the whole picture is resized to before it is cut (each slice that over the grid). Sizes are (height, width).
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub overview: (usize, usize),
    pub grid: Option<(usize, usize)>,
    pub refine: (usize, usize),
}

impl Plan {
    /// Each slice's size in pixels (none without a grid).
    pub fn slice(&self) -> (usize, usize) {
        match self.grid {
            Some((rows, cols)) => (self.refine.0 / rows, self.refine.1 / cols),
            None => (0, 0),
        }
    }

    /// The tiles' patch grids (rows, columns of 14-pixel patches): the overview's, then each slice's row by row.
    pub fn target_sizes(&self) -> Vec<(usize, usize)> {
        let mut sizes = vec![(self.overview.0 / PATCH, self.overview.1 / PATCH)];
        if let Some((rows, cols)) = self.grid {
            let (sh, sw) = self.slice();
            sizes.extend(std::iter::repeat((sh / PATCH, sw / PATCH)).take(rows * cols));
        }
        sizes
    }
}

/// The processor's plan for a picture of `h` x `w` pixels (`MiniCPMV4_6ImageProcessor._preprocess`).
pub fn plan(h: usize, w: usize, max_slices: usize) -> Plan {
    let grid = sliced_grid(h, w, max_slices, SCALE_RESOLUTION);
    let overview = best_resize(h as f64, w as f64, SCALE_RESOLUTION, grid.is_none());
    let refine = match grid {
        // `get_refine_size`: the picture's sides to multiples of the grid, a slice's share of them resized as a
        // tile is (up as well as down), times the grid
        Some((rows, cols)) => {
            let (rh, rw) = (ensure_divide(h as f64, rows), ensure_divide(w as f64, cols));
            let (bh, bw) = best_resize(rh as f64 / rows as f64, rw as f64 / cols as f64, SCALE_RESOLUTION, true);
            (bh * rows, bw * cols)
        }
        None => (0, 0),
    };
    Plan { overview, grid, refine }
}

/// The language model's tokens a tile of `(rows, cols)` patches becomes: one for each 4x4 patches (16x), or each 2x2
/// (4x, `merge` 2).
pub fn tile_tokens(size: (usize, usize), merge: usize) -> usize {
    (size.0 / merge) * (size.1 / merge)
}

/// The marker tokens around a picture's tiles, by id (`config.json`'s `image_start_id` ...; `newline_id` is `"\n"`).
#[derive(Clone, Copy, Debug)]
pub struct Markers {
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

impl Markers {
    fn has(&self, id: u32) -> bool {
        [self.slice_start, self.slice_end, self.image_start, self.image_end, self.newline].contains(&id)
    }
}

/// `torch.linspace(0, canvas - 1, n).round()`: `n` evenly spaced points over a canvas `canvas` long, each rounded
/// (halves to even).
fn spread(n: usize, canvas: usize) -> Vec<i64> {
    if n == 1 {
        return vec![0];
    }
    let step = (canvas as f64 - 1.) / (n as f64 - 1.);
    (0..n)
        .map(|i| {
            // (linspace's: the first half from the start, the second from the end, in f32)
            let v = if i < n / 2 { 0f32 + step as f32 * i as f32 } else { (canvas as f32 - 1.) - step as f32 * (n - 1 - i) as f32 };
            round_even(v as f64) as i64
        })
        .collect()
}

/// Every token's canvas M-RoPE position `[T, H, W]`, as `MiniCPMV4_7Model.get_rope_index` lays them out for one
/// sequence: `ids` the prompt's tokens, `image` whether each is a picture's (an `<|image_pad|>` the processor
/// expanded), `sizes` each tile's patch grid in the order the tiles come, `merge` 4 (16x) or 2 (4x). Text is counted
/// one by one; a picture's tokens and markers hold the time channel at its canvas's origin (the `<image>` marker's
/// place), rows and columns counted from it, the slices laid out over the canvas and the overview stretched over it;
/// the text after a picture goes on from the origin past the canvas's longer side. Returns the positions and the one
/// the next token takes.
pub fn canvas_positions(ids: &[u32], image: &[bool], sizes: &[(usize, usize)], merge: usize, m: Markers) -> (Vec<[i64; 3]>, i64) {
    let n = ids.len();
    let mut pos: Vec<[i64; 3]> = (0..n as i64).map(|i| [i; 3]).collect();
    // the crops: each maximal run of picture tokens; a frame is a crop not opened by `<slice>`, with the slices after it
    struct Frame {
        thumb: (usize, usize),
        slices: Vec<(usize, usize)>,
    }
    let mut groups: Vec<(usize, Vec<Frame>)> = Vec::new();
    let (mut i, mut previous_end) = (0, 0);
    while i < n {
        if !image[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && image[i] {
            i += 1;
        }
        let crop = (start, i);
        let opened_by_slice = start > 0 && ids[start - 1] == m.slice_start;
        match groups.last_mut().and_then(|g| g.1.last_mut()) {
            Some(frame) if opened_by_slice => frame.slices.push(crop),
            _ => {
                let gap = &ids[previous_end..start.saturating_sub(1)];
                let adjacent = !groups.is_empty() && gap.iter().all(|t| m.has(*t));
                let frame = Frame { thumb: crop, slices: Vec::new() };
                if adjacent {
                    groups.last_mut().unwrap().1.push(frame);
                } else {
                    groups.push((start - 1, vec![frame]));
                }
            }
        }
        previous_end = crop.1;
    }
    // a frame ends past the markers that close it
    let frame_end = |last: usize| {
        let mut end = last;
        while end < n && m.has(ids[end]) {
            end += 1;
        }
        end
    };
    let mut tiles = sizes.iter().copied();
    let (mut current, mut done) = (0i64, 0usize);
    for (group_start, frames) in &groups {
        let group_end = frame_end(frames.last().map(|f| f.slices.last().map_or(f.thumb.1, |s| s.1)).unwrap());
        for (k, p) in (done..*group_start).enumerate() {
            pos[p] = [current + k as i64; 3];
        }
        current += (*group_start - done) as i64;
        let mut cursor = *group_start;
        for frame in frames {
            let thumb_size = tiles.next().unwrap_or((0, 0));
            let start = frame.thumb.0 - 1;
            if start > cursor {
                for (k, p) in (cursor..start).enumerate() {
                    pos[p] = [current + k as i64; 3];
                }
                current += (start - cursor) as i64 + 1;
            }
            let end = frame_end(frame.slices.last().map_or(frame.thumb.1, |s| s.1)).min(group_end);
            let origin = current;
            let halo = (origin - 1).max(0);
            let (mut canvas_h, mut canvas_w) = ((thumb_size.0 / merge) as i64, (thumb_size.1 / merge) as i64);
            let (mut cols, mut slice_h, mut slice_w) = (0usize, 0i64, 0i64);
            let mut slice_sizes = Vec::new();
            if !frame.slices.is_empty() {
                // a gap wider than `</slice><slice>` is the newline ending a row: it fixes the columns
                cols = frame.slices.len();
                for k in 0..frame.slices.len() - 1 {
                    if frame.slices[k + 1].0 - frame.slices[k].1 > 2 {
                        cols = k + 1;
                        break;
                    }
                }
                let mut rows = frame.slices.len() / cols.max(1);
                if rows * cols != frame.slices.len() {
                    (rows, cols) = (1, frame.slices.len());
                }
                slice_sizes = frame.slices.iter().map(|_| tiles.next().unwrap_or((0, 0))).collect();
                (slice_h, slice_w) = ((slice_sizes[0].0 / merge) as i64, (slice_sizes[0].1 / merge) as i64);
                (canvas_h, canvas_w) = (rows as i64 * slice_h, cols as i64 * slice_w);
            }
            for p in pos.iter_mut().take(end).skip(start) {
                *p = [origin; 3];
            }
            pos[start] = [origin, halo, halo];
            if frame.thumb.1 < end {
                pos[frame.thumb.1] = [origin, origin + canvas_h, origin + canvas_w];
            }
            // the overview stretched over the canvas
            let (gh, gw) = (thumb_size.0 / merge, thumb_size.1 / merge);
            let (hs, ws) = (spread(gh, canvas_h as usize), spread(gw, canvas_w as usize));
            for (k, p) in (frame.thumb.0..frame.thumb.1).enumerate() {
                pos[p] = [origin, origin + hs[k / gw.max(1)], origin + ws[k % gw.max(1)]];
            }
            for (k, &(s0, s1)) in frame.slices.iter().enumerate() {
                let (h_off, w_off) = ((k / cols) as i64 * slice_h, (k % cols) as i64 * slice_w);
                let (sh, sw) = ((slice_sizes[k].0 / merge) as i64, (slice_sizes[k].1 / merge) as i64);
                if s0 >= start + 1 {
                    pos[s0 - 1] = [origin, origin + h_off, origin + w_off];
                }
                if s1 < end {
                    pos[s1] = [origin, origin + h_off + sh - 1, origin + w_off + sw - 1];
                }
                for (j, p) in (s0..s1).enumerate() {
                    pos[p] = [origin, origin + h_off + j as i64 / sw.max(1), origin + w_off + j as i64 % sw.max(1)];
                }
            }
            // the newline ending a row of slices: just past the row's right edge
            for k in 0..frame.slices.len().saturating_sub(1) {
                let (gap_start, gap_end) = (frame.slices[k].1, frame.slices[k + 1].0);
                if gap_end - gap_start <= 2 {
                    continue;
                }
                let boundary = ((k / cols) as i64 + 1) * slice_h - 1;
                for p in gap_start + 1..gap_end - 1 {
                    pos[p] = [origin, origin + boundary, origin + cols as i64 * slice_w];
                }
            }
            current = origin + canvas_h.max(canvas_w) + 1;
            cursor = end;
        }
        if cursor < group_end {
            for (k, p) in (cursor..group_end).enumerate() {
                pos[p] = [current + k as i64; 3];
            }
            current += (group_end - cursor) as i64;
        }
        done = group_end;
    }
    for (k, p) in (done..n).enumerate() {
        pos[p] = [current + k as i64; 3];
    }
    current += (n - done) as i64;
    (pos, current)
}

/// The window-attention merger after the tower's block `insert` (16x): attention within each 2x2 window of patches,
/// then the window's four as one (their concatenation through an MLP, tanh GELU, plus their mean).
#[derive(Debug)]
pub struct WindowMerger {
    pub insert: usize,
    pub ln1: (Tensor, Tensor),
    pub q: Weight,
    pub q_b: Tensor,
    pub k: Weight,
    pub k_b: Tensor,
    pub v: Weight,
    pub v_b: Tensor,
    pub o: Weight,
    pub o_b: Tensor,
    pub ds_ln: (Tensor, Tensor),
    pub up: Weight,
    pub up_b: Tensor,
    pub down: Weight,
    pub down_b: Tensor,
}

/// MiniCPM-V 4.7's tower and mergers (`MiniCPMV4_6VisionModel` and `MiniCPMV4_6Merger`): a SigLIP-400M tower (27
/// blocks, 1152 wide, no RoPE) over one tile at a time, its learned positions a 70 x 70 grid each tile's patches are
/// spread over; in 16x the window merger after block `insert`; then the final merger, each 2x2 of tokens concatenated,
/// normed and through an MLP (erf GELU) into the language model's width.
#[derive(Debug)]
pub struct MiniCpmVMmProj {
    pub config: MmProjConfig,
    pub patch_embd: Tensor,
    pub patch_embd_b: Tensor,
    /// `[70 * 70, D]` on the host: a tile's rows gathered from it.
    pub position_embd: Tensor,
    pub blocks: Vec<VisionBlock>,
    pub post_ln: (Tensor, Tensor),
    /// None: 4x (the final merger alone).
    pub window: Option<WindowMerger>,
    pub merger_ln: (Tensor, Tensor),
    pub merger_up: Weight,
    pub merger_up_b: Tensor,
    pub merger_down: Weight,
    pub merger_down_b: Tensor,
    /// The language model's own embeddings of `</image>`, `<slice>`, `</slice>` and `"\n"` (`v.tok_embd_sep`, `[4,
    /// lm]`), where the file keeps them.
    pub separators: Option<Tensor>,
    /// Patches a side per language-model token: 4 (16x) or 2 (4x).
    pub merge: usize,
    pub max_slices: usize,
    pub backend: Arc<dyn Backend>,
    /// The tower on the backend's device, once a tile has asked for it ([`crate::chain_minicpmv`]).
    pub chain: crate::chain_minicpmv::TowerCell,
}

/// A picture's tiles as the tower takes them: the overview, then the slices row by row, each `[3, H, W]` normalised.
pub struct Tiles {
    pub plan: Plan,
    pub tiles: Vec<Tensor>,
}

impl MiniCpmVMmProj {
    pub fn from_gguf_with_config(g: &GgufFile, config: MmProjConfig, backend: Arc<dyn Backend>) -> Result<Self> {
        let idx = TensorIndex::new(g);
        let int = |key: &str| g.metadata().get(key).and_then(|v| v.as_u64());
        let merge = int("clip.vision.projector.scale_factor").unwrap_or(4) as usize;
        if merge != 2 && merge != 4 {
            return Err(LlamaError::Config(format!("MiniCPM-V: projector scale_factor {merge}, not 2 (4x) or 4 (16x)")));
        }
        let insert = match g.metadata().get("clip.vision.wa_layer_indexes").and_then(|v| v.as_array()) {
            Some(gguf::Array::I32(v)) => v.first().map(|i| *i as usize),
            Some(gguf::Array::U32(v)) => v.first().map(|i| *i as usize),
            _ => None,
        }
        .unwrap_or(6);
        let max_slices = int("clip.vision.max_slice_nums").unwrap_or(MAX_SLICES as u64) as usize;
        let d = config.embedding_dim;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let t = |n: &str| idx.take(&format!("v.blk.{i}.{n}"), &[]);
            let w = |n: &str| idx.take_weight(&format!("v.blk.{i}.{n}"), &[]);
            blocks.push(VisionBlock {
                ln1_w: t("ln1.weight")?,
                ln1_b: t("ln1.bias")?,
                attn_q: w("attn_q.weight")?,
                attn_q_b: t("attn_q.bias")?,
                attn_k: w("attn_k.weight")?,
                attn_k_b: t("attn_k.bias")?,
                attn_v: w("attn_v.weight")?,
                attn_v_b: t("attn_v.bias")?,
                attn_out: w("attn_out.weight")?,
                attn_out_b: t("attn_out.bias")?,
                ln2_w: t("ln2.weight")?,
                ln2_b: t("ln2.bias")?,
                ffn_up: w("ffn_up.weight")?,
                ffn_up_b: t("ffn_up.bias")?,
                ffn_down: w("ffn_down.weight")?,
                ffn_down_b: t("ffn_down.bias")?,
            });
        }
        // (16x only: the 4x file has no window merger)
        let window = if merge == 4 {
            let t = |n: &str| idx.take(&format!("v.vit_merger.{n}"), &[]);
            let w = |n: &str| idx.take_weight(&format!("v.vit_merger.{n}"), &[]);
            if insert + 1 >= config.n_layers {
                return Err(LlamaError::Config(format!("MiniCPM-V: the window merger after block {insert} of {}", config.n_layers)));
            }
            Some(WindowMerger {
                insert,
                ln1: (t("ln1.weight")?, t("ln1.bias")?),
                q: w("attn_q.weight")?,
                q_b: t("attn_q.bias")?,
                k: w("attn_k.weight")?,
                k_b: t("attn_k.bias")?,
                v: w("attn_v.weight")?,
                v_b: t("attn_v.bias")?,
                o: w("attn_out.weight")?,
                o_b: t("attn_out.bias")?,
                ds_ln: (t("ds_ln.weight")?, t("ds_ln.bias")?),
                up: w("ds_ffn_up.weight")?,
                up_b: t("ds_ffn_up.bias")?,
                down: w("ds_ffn_down.weight")?,
                down_b: t("ds_ffn_down.bias")?,
            })
        } else {
            None
        };
        let m = Self {
            patch_embd: idx.take("v.patch_embd.weight", &[])?,
            patch_embd_b: idx.take("v.patch_embd.bias", &[])?,
            position_embd: idx.take("v.position_embd.weight", &[])?,
            blocks,
            post_ln: (idx.take("v.post_ln.weight", &[])?, idx.take("v.post_ln.bias", &[])?),
            window,
            merger_ln: (idx.take("mm.input_norm.weight", &[])?, idx.take("mm.input_norm.bias", &[])?),
            merger_up: idx.take_weight("mm.up.weight", &[])?,
            merger_up_b: idx.take("mm.up.bias", &[])?,
            merger_down: idx.take_weight("mm.down.weight", &[])?,
            merger_down_b: idx.take("mm.down.bias", &[])?,
            separators: idx.try_take("v.tok_embd_sep").transpose()?,
            merge,
            max_slices,
            config,
            backend: backend.clone(),
            chain: Default::default(),
        };
        // the shapes the forward pass assumes
        let bad = |name: &str, got: &[usize], want: &[usize]| LlamaError::BadTensorShape {
            name: name.into(),
            got: got.iter().map(|&v| v as u64).collect(),
            expected: want.iter().map(|&v| v as u64).collect(),
        };
        let p = m.config.patch_size;
        if m.patch_embd.shape() != [d, 3, p, p] {
            return Err(bad("v.patch_embd.weight", m.patch_embd.shape(), &[d, 3, p, p]));
        }
        if m.position_embd.shape() != [70 * 70, d] {
            return Err(bad("v.position_embd.weight", m.position_embd.shape(), &[70 * 70, d]));
        }
        if m.merger_up.shape() != [4 * d, 4 * d] {
            return Err(bad("mm.up.weight", m.merger_up.shape(), &[4 * d, 4 * d]));
        }
        if let Some(w) = &m.window {
            if w.up.shape().get(1) != Some(&(4 * d)) || w.down.shape().first() != Some(&d) {
                return Err(bad("v.vit_merger.ds_ffn_up.weight", w.up.shape(), &[0, 4 * d]));
            }
        }
        if let Some(s) = &m.separators {
            if s.shape() != [4, m.lm_width()] {
                return Err(bad("v.tok_embd_sep", s.shape(), &[4, m.lm_width()]));
            }
        }
        // everything but the host-side tables onto the backend (one-shot a picture: a weight left on the host only
        // slows a picture)
        const MARGIN: usize = 1 << 30;
        let up = |t: Tensor| backend.to_device(t);
        let w = |x: Weight| x.try_to_device(&*backend, MARGIN);
        Ok(Self {
            blocks: m
                .blocks
                .into_iter()
                .map(|b| VisionBlock {
                    ln1_w: up(b.ln1_w),
                    ln1_b: up(b.ln1_b),
                    attn_q: w(b.attn_q),
                    attn_q_b: up(b.attn_q_b),
                    attn_k: w(b.attn_k),
                    attn_k_b: up(b.attn_k_b),
                    attn_v: w(b.attn_v),
                    attn_v_b: up(b.attn_v_b),
                    attn_out: w(b.attn_out),
                    attn_out_b: up(b.attn_out_b),
                    ln2_w: up(b.ln2_w),
                    ln2_b: up(b.ln2_b),
                    ffn_up: w(b.ffn_up),
                    ffn_up_b: up(b.ffn_up_b),
                    ffn_down: w(b.ffn_down),
                    ffn_down_b: up(b.ffn_down_b),
                })
                .collect(),
            window: m.window.map(|x| WindowMerger { q: w(x.q), k: w(x.k), v: w(x.v), o: w(x.o), up: w(x.up), down: w(x.down), ..x }),
            merger_up: w(m.merger_up),
            merger_down: w(m.merger_down),
            ..m
        })
    }

    /// The language model's width (the final merger's output).
    pub fn lm_width(&self) -> usize {
        self.merger_down.shape()[0]
    }

    /// A picture's tiles, as [`tiles`] cuts it for this file's tower.
    pub fn tiles(&self, picture: &image::RgbImage) -> Tiles {
        tiles(picture, self.max_slices, self.config.mean, self.config.std)
    }

    /// One tile `[3, H, W]` (sides multiples of four patches) to its tokens in the language model's space, row-major
    /// over its `H / (14 merge)` x `W / (14 merge)` grid.
    pub fn forward(&self, tile: &Tensor) -> Result<Tensor> {
        let cfg = &self.config;
        let (p, d) = (cfg.patch_size, cfg.embedding_dim);
        let shape = tile.shape();
        if shape.len() != 3 || shape[0] != 3 || shape[1] % (p * 4) != 0 || shape[2] % (p * 4) != 0 {
            return Err(LlamaError::Config(format!("MiniCPM-V: a tile of {shape:?}, not [3, H, W] in multiples of {}", p * 4)));
        }
        // chained on the backend's GPU where it has one
        if let Some(out) = crate::chain_minicpmv::forward(self, &self.chain, tile) {
            return Ok(out);
        }
        let (mut gh, mut gw) = (shape[1] / p, shape[2] / p);
        let b = &*self.backend;
        // patches through the patch convolution (a matmul), plus each one's learned position: the 70 x 70 grid's
        // nearest (`get_vision_nearest_position_ids`: floor(70 i / rows) is torch's bucketize here, every size to 300)
        let patches = b.to_device(unfold_patches_to_host(tile, p)?);
        let weight = b.to_device(self.patch_embd.to_host().reshape(vec![d, 3 * p * p]).map_err(|e| LlamaError::Config(format!("patch_embd: {e:?}")))?);
        let mut x = b.linear(&patches, &weight);
        add_bias_inplace(b, &mut x, &self.patch_embd_b);
        let table = self.position_embd.to_host();
        let mut pos = vec![0f32; gh * gw * d];
        for i in 0..gh {
            for j in 0..gw {
                let id = (70 * i / gh) * 70 + 70 * j / gw;
                pos[(i * gw + j) * d..][..d].copy_from_slice(&table.data()[id * d..][..d]);
            }
        }
        b.add_inplace(&mut x, &b.to_device(Tensor::from_vec(pos, vec![gh * gw, d])));
        for (i, blk) in self.blocks.iter().enumerate() {
            x = vision_block_forward(b, &x, blk, cfg);
            if let Some(w) = self.window.as_ref().filter(|w| w.insert == i) {
                x = self.window_merge(w, &x, gh, gw);
                (gh, gw) = (gh / 2, gw / 2);
            }
        }
        x = layer_norm(b, &x, &self.post_ln.0, &self.post_ln.1, cfg.layer_norm_eps);
        // the final merger: each 2x2's four rows side by side, normed, up, erf GELU, down
        let cat = b.to_device(gather_quads(&x.to_host(), gh, gw, d));
        let mut h = layer_norm(b, &cat, &self.merger_ln.0, &self.merger_ln.1, cfg.layer_norm_eps);
        h = self.merger_up.linear(b, &h);
        b.add_inplace_broadcast_last(&mut h, &self.merger_up_b);
        h = qwen_merger_gelu(b, &h);
        let mut out = self.merger_down.linear(b, &h);
        b.add_inplace_broadcast_last(&mut out, &self.merger_down_b);
        Ok(out)
    }

    /// The window merger over a `gh` x `gw` grid: attention within each 2x2 window (on the host: four tokens a
    /// window), added back, then each window's four to one.
    fn window_merge(&self, w: &WindowMerger, x: &Tensor, gh: usize, gw: usize) -> Tensor {
        let b = &*self.backend;
        let cfg = &self.config;
        let (d, heads, hd) = (cfg.embedding_dim, cfg.n_heads, cfg.head_dim);
        let h0 = layer_norm(b, x, &w.ln1.0, &w.ln1.1, cfg.layer_norm_eps);
        let project = |m: &Weight, bias: &Tensor| {
            let mut y = m.linear(b, &h0);
            b.add_inplace_broadcast_last(&mut y, bias);
            y.to_host()
        };
        let (q, k, v) = (project(&w.q, &w.q_b), project(&w.k, &w.k_b), project(&w.v, &w.v_b));
        let (q, k, v) = (q.data(), k.data(), v.data());
        let scale = 1. / (hd as f32).sqrt();
        let mut att = vec![0f32; gh * gw * d];
        for wi in 0..gh / 2 {
            for wj in 0..gw / 2 {
                let rows = [(2 * wi) * gw + 2 * wj, (2 * wi) * gw + 2 * wj + 1, (2 * wi + 1) * gw + 2 * wj, (2 * wi + 1) * gw + 2 * wj + 1];
                for hh in 0..heads {
                    fn head(t: &[f32], r: usize, d: usize, off: usize, hd: usize) -> &[f32] {
                        &t[r * d + off..][..hd]
                    }
                    let off = hh * hd;
                    for &r in &rows {
                        let scores: Vec<f32> = rows.iter().map(|&s| head(q, r, d, off, hd).iter().zip(head(k, s, d, off, hd)).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
                        let most = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let e: Vec<f32> = scores.iter().map(|s| (s - most).exp()).collect();
                        let sum: f32 = e.iter().sum();
                        let out = &mut att[r * d + hh * hd..][..hd];
                        for (&s, &p) in rows.iter().zip(&e) {
                            for (o, val) in out.iter_mut().zip(head(v, s, d, off, hd)) {
                                *o += p / sum * val;
                            }
                        }
                    }
                }
            }
        }
        let mut y = w.o.linear(b, &b.to_device(Tensor::from_vec(att, vec![gh * gw, d])));
        b.add_inplace_broadcast_last(&mut y, &w.o_b);
        b.add_inplace(&mut y, x);
        // each window's four: their concatenation through the MLP, plus their mean
        let host = y.to_host();
        let cat = gather_quads(&host, gh, gw, d);
        let n = (gh / 2) * (gw / 2);
        let mut mean = vec![0f32; n * d];
        for (m, quad) in mean.chunks_exact_mut(d).zip(cat.data().chunks_exact(4 * d)) {
            for (j, o) in m.iter_mut().enumerate() {
                *o = (quad[j] + quad[d + j] + quad[2 * d + j] + quad[3 * d + j]) * 0.25;
            }
        }
        let mut h = layer_norm(b, &b.to_device(cat), &w.ds_ln.0, &w.ds_ln.1, cfg.layer_norm_eps);
        h = w.up.linear(b, &h);
        b.add_inplace_broadcast_last(&mut h, &w.up_b);
        h = b.gelu_approx(&h);
        let mut out = w.down.linear(b, &h);
        b.add_inplace_broadcast_last(&mut out, &w.down_b);
        b.add_inplace(&mut out, &b.to_device(Tensor::from_vec(mean, vec![n, d])));
        out
    }
}

/// A picture's tiles: cut as the processor cuts it ([`plan`], at most `max_slices` slices), each resized from the
/// picture itself (bicubic, antialiased as torchvision's is, to 8 bits as it is) and normalised by `mean` and `std`.
pub fn tiles(picture: &image::RgbImage, max_slices: usize, mean: [f32; 3], std: [f32; 3]) -> Tiles {
    let (w, h) = picture.dimensions();
    let plan = plan(h as usize, w as usize, max_slices);
    let normalise = |img: &image::RgbImage| -> Tensor {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut data = vec![0f32; 3 * h * w];
        for (i, px) in img.pixels().enumerate() {
            for c in 0..3 {
                data[c * h * w + i] = (px.0[c] as f32 / 255. - mean[c]) / std[c];
            }
        }
        Tensor::from_vec(data, vec![3, h, w])
    };
    let resize = |(th, tw): (usize, usize)| image::imageops::resize(picture, tw as u32, th as u32, image::imageops::FilterType::CatmullRom);
    let mut tiles = vec![normalise(&resize(plan.overview))];
    if let Some((rows, cols)) = plan.grid {
        let refine = resize(plan.refine);
        let (sh, sw) = plan.slice();
        for r in 0..rows {
            for c in 0..cols {
                let cut = image::imageops::crop_imm(&refine, (c * sw) as u32, (r * sh) as u32, sw as u32, sh as u32).to_image();
                tiles.push(normalise(&cut));
            }
        }
    }
    Tiles { plan, tiles }
}

/// [`tiles`] of an encoded picture (PNG, JPEG, WebP).
pub fn tiles_from_bytes(bytes: &[u8], max_slices: usize, mean: [f32; 3], std: [f32; 3]) -> Result<Tiles> {
    let picture = image::load_from_memory(bytes).map_err(|e| LlamaError::Config(format!("picture: {e}")))?.to_rgb8();
    Ok(tiles(&picture, max_slices, mean, std))
}

/// A `gh` x `gw` grid's rows (`[gh * gw, d]`) as its 2x2 blocks', row-major, each block's four side by side: top left,
/// top right, bottom left, bottom right (`[gh/2 * gw/2, 4 d]`).
fn gather_quads(x: &Tensor, gh: usize, gw: usize, d: usize) -> Tensor {
    let data = x.data();
    let (oh, ow) = (gh / 2, gw / 2);
    let mut out = vec![0f32; oh * ow * 4 * d];
    for i in 0..oh {
        for j in 0..ow {
            let dst = &mut out[(i * ow + j) * 4 * d..][..4 * d];
            for (q, (di, dj)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                let r = (2 * i + di) * gw + 2 * j + dj;
                dst[q * d..][..d].copy_from_slice(&data[r * d..][..d]);
            }
        }
    }
    Tensor::from_vec(out, vec![oh * ow, 4 * d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use oaiy_engine::json::Json;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    /// A safetensors file's F32 and I64 tensors (`tools/minicpmv/reference.py`'s), each its shape and values as f32.
    fn reference(path: &Path) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let bytes = std::fs::read(path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = Json::parse(&bytes[8..8 + n]).unwrap();
        let data = &bytes[8 + n..];
        let mut out = BTreeMap::new();
        for (name, t) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = t.get("shape").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect();
            let off: Vec<usize> = t.get("data_offsets").unwrap().as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect();
            let raw = &data[off[0]..off[1]];
            let values = match t.get("dtype").and_then(Json::as_str) {
                Some("F32") => raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                Some("I64") => raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
                other => panic!("{name}: {other:?}"),
            };
            out.insert(name.clone(), (shape, values));
        }
        out
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b))
    }

    /// The canvas positions of transformers' own prompt (`OAIY_MINICPMV_REFERENCE`: its tokens, which are a picture's,
    /// its tiles' grids) are its `get_rope_index`'s, every token's exactly, and the reply goes on from one past them.
    #[test]
    #[ignore = "needs transformers' run on a picture (OAIY_MINICPMV_REFERENCE, tools/minicpmv/reference.py)"]
    fn canvas_positions_are_transformers() {
        let Some(want) = std::env::var_os("OAIY_MINICPMV_REFERENCE").map(PathBuf::from) else { return };
        let want = reference(&want);
        let ids: Vec<u32> = want["input_ids"].1.iter().map(|v| *v as u32).collect();
        let image: Vec<bool> = want["mm_token_type_ids"].1.iter().map(|v| *v != 0.).collect();
        let sizes: Vec<(usize, usize)> = want["target_sizes"].1.chunks_exact(2).map(|s| (s[0] as usize, s[1] as usize)).collect();
        // (MiniCPM-V 4.7's config.json: image_start_id ... newline_id)
        let markers = Markers { image_start: 248078, image_end: 248079, slice_start: 248088, slice_end: 248089, newline: 198 };
        let (_, tiles) = &want["features.0"];
        let merge = if tiles.len() / 1024 == tile_tokens(sizes[0], 4) || tiles.len() / 2048 == tile_tokens(sizes[0], 4) { 4 } else { 2 };
        let (pos, next) = canvas_positions(&ids, &image, &sizes, merge, markers);
        let (_, theirs) = &want["position_ids"];
        let n = ids.len();
        let mut wrong = 0;
        for (i, p) in pos.iter().enumerate() {
            let t = [theirs[i] as i64, theirs[n + i] as i64, theirs[2 * n + i] as i64];
            if *p != t {
                wrong += 1;
                if wrong < 6 {
                    eprintln!("token {i} ({}): {p:?}, transformers' {t:?}", ids[i]);
                }
            }
        }
        let most = (0..3 * n).map(|i| theirs[i] as i64).max().unwrap();
        eprintln!("{n} tokens ({} a picture's), {wrong} placed otherwise; the reply from {next} (transformers' {})", image.iter().filter(|x| **x).count(), most + 1);
        assert_eq!((wrong, next), (0, most + 1));
    }

    /// transformers' run on a picture (`OAIY_MINICPMV_REFERENCE`: `tools/minicpmv/reference.py`'s output, the mode the
    /// projector `OAIY_MINICPMV_MMPROJ` is for) against this one on the CPU: the same tiles (the plan exactly; their
    /// pixels within the resize filters' difference), and each of transformers' tiles to tokens alike.
    #[test]
    #[ignore = "needs MiniCPM-V 4.7's projector (OAIY_MINICPMV_MMPROJ), the picture (OAIY_MINICPMV_PICTURE) and transformers' run on it (OAIY_MINICPMV_REFERENCE)"]
    fn tiles_and_their_tokens_are_transformers() {
        let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
        let (Some(mmproj), Some(picture), Some(want)) = (var("OAIY_MINICPMV_MMPROJ"), var("OAIY_MINICPMV_PICTURE"), var("OAIY_MINICPMV_REFERENCE")) else { return };
        let want = reference(&want);
        let g = gguf::GgufFile::open(&mmproj).unwrap();
        let MmProj::MiniCpmV(mm) = MmProj::from_gguf(&g, Arc::new(ggml_rs::CpuBackend::new())).unwrap() else { panic!("not MiniCPM-V") };
        let picture = image::open(&picture).unwrap().to_rgb8();
        let tiles = mm.tiles(&picture);
        let sizes: Vec<(usize, usize)> = want["target_sizes"].1.chunks_exact(2).map(|s| (s[0] as usize, s[1] as usize)).collect();
        assert_eq!(tiles.plan.target_sizes(), sizes);
        for (k, tile) in tiles.tiles.iter().enumerate() {
            let (shape, theirs) = &want[&format!("tile.{k}")];
            assert_eq!(tile.shape(), shape.as_slice(), "tile {k}");
            let host = tile.to_host();
            let diff: Vec<f32> = host.data().iter().zip(theirs).map(|(a, b)| (a - b).abs()).collect();
            let mean = diff.iter().sum::<f32>() / diff.len() as f32;
            let worst = diff.iter().copied().fold(0., f32::max);
            eprintln!("tile {k} {shape:?}: pixels off by {mean:.4} on average, at most {worst:.3} (of 2)");
            assert!(mean < 0.02, "tile {k}: {mean}");
            // the tower on transformers' own pixels
            let (_, pixels) = &want[&format!("tile.{k}")];
            let out = mm.forward(&Tensor::from_vec(pixels.clone(), shape.clone())).unwrap().to_host();
            let (fshape, features) = &want[&format!("features.{k}")];
            assert_eq!(out.shape(), fshape.as_slice(), "tile {k}");
            let c = cosine(out.data(), features);
            let rms = (features.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / features.len() as f64).sqrt();
            let worst = out.data().iter().zip(features).map(|(a, b)| (a - b).abs() as f64).fold(0., f64::max);
            let ours = cosine(mm.forward(tile).unwrap().to_host().data(), features);
            eprintln!("tile {k}: tokens {fshape:?}, cosine {c:.6} (worst {worst:.3} of an RMS {rms:.3}); from our pixels {ours:.5}");
            assert!(c > 0.999, "tile {k}: cosine {c}");
        }
    }

    /// The processor's grids and sizes: a square megapixel picture is two rows of three (its log ratios not quite a
    /// tie in f64: |ln(2/3)| is one ulp past ln(3/2)), a small one is one tile scaled up, sides multiples of 56 pixels.
    #[test]
    fn pictures_are_cut_as_the_processor_cuts_them() {
        let square = plan(1024, 1024, 9);
        assert_eq!(square.grid, Some((2, 3)));
        assert_eq!(square.overview, (448, 448));
        let small = plan(300, 200, 9);
        assert_eq!(small.grid, None);
        assert_eq!((small.overview.0 % 56, small.overview.1 % 56), (0, 0));
        for (h, w) in [(1344, 800), (512, 1024), (4000, 3000), (100, 2000)] {
            let p = plan(h, w, 9);
            let (sh, sw) = p.slice();
            assert!(p.overview.0 % 56 == 0 && p.overview.1 % 56 == 0 && sh % 56 == 0 && sw % 56 == 0, "{h}x{w}: {p:?}");
        }
        assert_eq!(spread(4, 7), vec![0, 2, 4, 6]);
        assert_eq!(spread(3, 4), vec![0, 2, 3]);
    }
}
