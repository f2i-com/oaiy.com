//! A decode step's expert on the host as [`Exl3Slots`] makes one on a card, to the bit: the input transform, the
//! one-row kernel's sums in that kernel's order (a split's tiles a workgroup's eight warps in turn, each lane's two
//! columns its own sums, the warps' and lanes' added in order, the splits' added by the output transform), the output
//! transform, and between gate and up and down the SwiGLU from one table both sides read ([`silu_table`]). So an expert
//! the cards do not hold is made by the host's cores from its record in RAM, where its upload was a millisecond or two
//! of a card's link, and its output is the one a card would have given: what a reply says does not hang on which
//! experts the cards held when it was asked.

use super::*;

/// The splits of a one-row projection's tile rows a card's slots make (`Exl3Slots`' groups: enough workgroups to fill
/// a large GPU, at most 8): its partial sums are added in this many parts, so the host's must be too.
pub fn step_splits(k: usize, n: usize) -> usize {
    4096usize.div_ceil(n / 16).next_power_of_two().min(8).min(k / 16).max(1)
}

/// The SwiGLU's `silu(g)` of every value a step's gate output can take (an f16's, by its bits: the output transform
/// rounds there), `g` taken as `min(g, limit)` where a limit is given (> 0): `g / (1 + e^-g)` in f32, made here once
/// for each limit. A card's slots look it up as the host does: each made its own exponential before, the card's its
/// own approximation, and the two then differed in an f16 now and then.
pub fn silu_table(limit: f32) -> Arc<Vec<f32>> {
    static MADE: Mutex<Vec<(u32, Arc<Vec<f32>>)>> = Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, t)) = made.iter().find(|(l, _)| *l == limit.to_bits()) {
        return Arc::clone(t);
    }
    let table: Arc<Vec<f32>> = Arc::new(
        (0..=u16::MAX)
            .map(|bits| {
                let g = half::f16::from_bits(bits).to_f32();
                let g = if limit > 0.0 { g.min(limit) } else { g };
                g / (1.0 + (-g).exp())
            })
            .collect(),
    );
    made.push((limit.to_bits(), Arc::clone(&table)));
    table
}

/// Whether this host makes a step's experts quickly ([`step_experts_host`] on its vectors: some half a millisecond an
/// expert on 16 cores of AVX-512, where a card took a record in over its link at one to two and a half), so a model
/// may make the experts its cards do not hold here.
pub fn step_host_fast() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        avx512::available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// An f16 value's bits (`v` one: a step's output transform rounds there).
#[inline]
pub(super) fn f16_bits(v: f32) -> usize {
    half::f16::from_f32(v).to_bits() as usize
}

impl ProjAt {
    /// The one-row kernel's sums of tile columns `cols` (16 outputs each) of this projection of `record` for a
    /// prepared row `xh` ([`Self::prepare`]'s), as a card's slots make them ([`step_splits`]' parts, each summed as
    /// that kernel sums, then added in order): `[16 Â· cols]`, for [`Self::finish`] once every column's are there.
    /// `fused`: a lane's products added by multiply-adds (one rounding each), as the card's compiler makes them.
    pub fn step_sums(&self, record: &[u8], xh: &[f32], cols: std::ops::Range<usize>, fused: bool) -> Vec<f32> {
        let mut sums = vec![0f32; cols.len() * 16];
        for s in 0..step_splits(self.k, self.n) {
            for (a, b) in sums.iter_mut().zip(self.split_sums(record, xh, s, cols.clone(), fused)) {
                *a += b;
            }
        }
        sums
    }

    /// Split `split`'s part of [`Self::step_sums`] (its tile rows' sums of tile columns `cols`, `[16 Â· cols]`), which
    /// the splits' added in order make: a split's tile rows lie together in the record, so the host's cores read them
    /// as they lie, a task a split (a column's tile rows apart, a tile a row, were reads 14 KB apart).
    pub fn split_sums(&self, record: &[u8], xh: &[f32], split: usize, cols: std::ops::Range<usize>, fused: bool) -> Vec<f32> {
        let trellis = &record[self.trellis..self.trellis + self.trellis_bytes()];
        let kts = self.k / 16;
        let per = kts.div_ceil(step_splits(self.k, self.n));
        let rows = (split * per).min(kts)..((split + 1) * per).min(kts);
        #[cfg(target_arch = "x86_64")]
        if avx512::available() && self.tile_words <= 64 {
            // SAFETY: the CPU has the features the function is compiled for (checked at run time just above).
            return unsafe { avx512::split_sums(trellis, self.k, self.n, self.tile_words, xh, rows, cols) };
        }
        split_sums(&LeWords(trellis), self.k, self.n, self.tile_words, xh, rows, cols, fused)
    }

    /// The SwiGLU of a step's gate and up outputs (`[k]` each, as [`Self::finish`] gives them) as this projection's
    /// input transform reads it on a card: up clamped at `limit` (> 0), `silu` of the gate from `table`
    /// ([`silu_table`]'s for that limit), their product rounded to f16, then scaled, transformed and rounded as
    /// [`Self::prepare`] does a row.
    pub fn prepare_swiglu(&self, record: &[u8], gate: &[f32], up: &[f32], limit: f32, table: &[f32]) -> Vec<f32> {
        let scale = self.input_scales(record);
        let mut out: Vec<f32> = gate
            .iter()
            .zip(up)
            .enumerate()
            .map(|(i, (&g, &u))| {
                let u = if limit > 0.0 { u.clamp(-limit, limit) } else { u };
                half(table[f16_bits(g)] * u) * scale(i)
            })
            .collect();
        transformed(&mut out);
        out
    }

    /// One row as [`Self::prepare`] makes it of a record (whose input map is the identity), its scales read from the
    /// record as they are used, its roundings to f16 sixteen at a time where the CPU has AVX-512 (each value's own
    /// conversion was most of a step's input's time).
    pub fn prepare_row(&self, record: &[u8], x: &[f32]) -> Vec<f32> {
        let scale = self.input_scales(record);
        let mut out = vec![0f32; self.k];
        scaled_halves(&x[..self.k], &record[self.suh..self.suh + 2 * self.k], &mut out, &scale);
        transformed(&mut out);
        out
    }

    /// The input's scales (`suh`, f16 values in the record), by index; the record's codebook checked first.
    fn input_scales<'r>(&self, record: &'r [u8]) -> impl Fn(usize) -> f32 + 'r {
        assert_eq!(record[self.mul1..self.mul1 + 4], MUL1, "an expert's projection of a codebook other than mul1");
        let at = self.suh;
        move |i| half::f16::from_bits(u16::from_le_bytes([record[at + 2 * i], record[at + 2 * i + 1]])).to_f32()
    }
}

/// A scaled input row transformed and rounded as [`ProjAt::prepare`] makes it: the Hadamard transform of each block of
/// 128, then each value scaled by 1/sqrt(128) and rounded to f16.
fn transformed(out: &mut [f32]) {
    had(out);
    scaled_by(out, ISQRT128, None);
}

/// `out[i] = half(x[i]) * scale(i)` (`scales` the same scales as f16 bytes): a row's input as [`ProjAt::prepare`]
/// scales it, sixteen at a time where the CPU has AVX-512 (the same roundings: a conversion to f16 rounds to nearest,
/// ties to even, as `half` does).
fn scaled_halves(x: &[f32], scales: &[u8], out: &mut [f32], scale: &dyn Fn(usize) -> f32) {
    #[cfg(target_arch = "x86_64")]
    if avx512::available() {
        // SAFETY: the CPU has AVX-512 F (checked just above); the lengths are checked inside.
        return unsafe { avx512::scaled_halves(x, scales, out) };
    }
    for (i, (o, &v)) in out.iter_mut().zip(x).enumerate() {
        *o = half(v) * scale(i);
    }
}

/// `v[i] = half(v[i] * by * s[i])`, `s` the f16 scales `scales` holds (none: 1, `half(v[i] * by)`): an output's or an
/// input's last rounding, sixteen at a time where the CPU has AVX-512, the products in the same order.
fn scaled_by(v: &mut [f32], by: f32, scales: Option<&[u8]>) {
    #[cfg(target_arch = "x86_64")]
    if avx512::available() {
        // SAFETY: the CPU has AVX-512 F (checked just above); the lengths are checked inside.
        return unsafe { avx512::scaled_by(v, by, scales) };
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = match scales {
            Some(s) => half(*x * by * half::f16::from_bits(u16::from_le_bytes([s[2 * i], s[2 * i + 1]])).to_f32()),
            None => half(*x * by),
        };
    }
}

/// A projection's one-row sums of tile columns `cols` as the one-row kernel makes them on a card (see
/// [`ProjAt::step_sums`]): `splits` parts of the tile rows ([`split_sums`] each), added in order. The tests' reference
/// (the scalar emulation whatever the host has).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn one_row_sums<W: Words + ?Sized>(words: &W, k: usize, n: usize, tile_words: usize, xh: &[f32], splits: usize, cols: std::ops::Range<usize>, fused: bool) -> Vec<f32> {
    let (kts, mut out) = (k / 16, vec![0f32; cols.len() * 16]);
    let per = kts.div_ceil(splits);
    for s in 0..splits {
        for (a, b) in out.iter_mut().zip(split_sums(words, k, n, tile_words, xh, (s * per).min(kts)..((s + 1) * per).min(kts), cols.clone(), fused)) {
            *a += b;
        }
    }
    out
}

/// A split's part of [`one_row_sums`] (its tile `rows`; a card's workgroup of eight warps, each taking the split's
/// tiles in turn, a lane's two columns' sums its own): each column's sum of its warps' four lanes' sums, the warps in
/// order. A whole number of bits a weight (lane `l`'s eight codes in one 64-bit window, as [`tile_columns_lanes`]
/// reads them). The tiles taken as they lie (tile row by tile row), each warp's sums kept for each column.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_sums<W: Words + ?Sized>(words: &W, k: usize, n: usize, tile_words: usize, xh: &[f32], rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, fused: bool) -> Vec<f32> {
    assert!(tile_words % 16 == 0 && (16..=96).contains(&tile_words), "a step's sums of a whole number of bits a weight, up to 6");
    let (ntiles, nw, b) = (n / 16, tile_words / 2, tile_words / 16);
    assert!(cols.end <= ntiles && rows.end <= k / 16 && xh.len() >= k, "a step's sums of tile columns {cols:?} of {ntiles}");
    let bits = nw * 32;
    let book = by_bytesum();
    let starts: Vec<(usize, u32)> = (0..32)
        .map(|l| {
            let first = (b * (8 * l + 1) + bits - 16) % bits;
            (first / 32, (first % 32) as u32)
        })
        .collect();
    let mad = |acc: f32, x: f32, w: f32| if fused { x.mul_add(w, acc) } else { acc + x * w };
    // each column's warps' lanes' two sums (columns l / 4 and l / 4 + 8 of the tile column)
    let mut lo = vec![[[0f32; 32]; 8]; cols.len()];
    let mut hi = vec![[[0f32; 32]; 8]; cols.len()];
    let mut tile = [0u32; 64];
    for kt in rows.clone() {
        let w = (kt - rows.start) % 8;
        let x = &xh[kt * 16..kt * 16 + 16];
        for nt in cols.clone() {
            let base = (kt * ntiles + nt) * nw;
            for (i, t) in tile[..nw].iter_mut().enumerate() {
                *t = words.word(base + i);
            }
            let (lo, hi) = (&mut lo[nt - cols.start][w], &mut hi[nt - cols.start][w]);
            for (l, &(q, sh)) in starts.iter().enumerate() {
                let (a, bb, c) = (tile[q] as u64, tile[(q + 1) % nw] as u64, tile[(q + 2) % nw] as u64);
                let win = ((a << 32 | bb) << sh) | ((c << sh) >> 32);
                let rb = 2 * (l % 4);
                let xv = [x[rb], x[rb + 1], x[rb + 8], x[rb + 9]];
                let value = |j: usize| book[bytesum(((win >> (48 - b * j)) & 0xffff) as u32)];
                let (mut a0, mut a1) = (lo[l], hi[l]);
                for (j, &xj) in xv.iter().enumerate() {
                    a0 = mad(a0, xj, value(j));
                }
                for (j, &xj) in xv.iter().enumerate() {
                    a1 = mad(a1, xj, value(j + 4));
                }
                lo[l] = a0;
                hi[l] = a1;
            }
        }
    }
    column_sums(&lo, &hi)
}

/// Each tile column's 16 sums of its warps' lanes' (`lo`, `hi`: each column's warps' lanes' two sums), as the card's
/// workgroup adds them: a column's four lanes of each warp, the warps in order.
fn column_sums(lo: &[[[f32; 32]; 8]], hi: &[[[f32; 32]; 8]]) -> Vec<f32> {
    let mut out = vec![0f32; lo.len() * 16];
    for (col, sums) in out.chunks_exact_mut(16).enumerate() {
        for (c, sum) in sums.iter_mut().enumerate() {
            let from = if c < 8 { &lo[col] } else { &hi[col] };
            let mut total = 0.0f32;
            for warp in from {
                for q in 0..4 {
                    total += warp[(c % 8) * 4 + q];
                }
            }
            *sum = total;
        }
    }
    out
}
/// [`split_sums`] on AVX-512's vectors (with its byte dot products, VNNI's): a vector a half of a warp's 32 lanes,
/// each vector lane one GPU lane's, so each sum is that lane's in its order (the products are exact in f32, an f16
/// input's by an f16 weight's, so a multiply-add is the card's multiply and add); a code's 16 bits found by two
/// permutes of the tile's words and two shifts by amounts made once for the bitrate, its value in mul1's closed form
/// (the byte sum a dot product, the one rounding to f16 the conversion's, to nearest even). The warps' and lanes' sums
/// added as the card adds them. Some ten times the scalar emulation's speed.
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

    /// Whether this CPU has the vectors [`split_sums`] is compiled for (OAIY_EXL3_STEP_SCALAR: the scalar
    /// emulation whatever it has, to compare against).
    pub(super) fn available() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            std::env::var_os("OAIY_EXL3_STEP_SCALAR").is_none()
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
        })
    }

    /// Where a bitrate's codes lie for each half `h` of a warp's lanes and code `j`: each vector lane's word (the
    /// tile's, 0 to 31), the word after it, and how far into the first the code starts.
    struct Plan {
        first: [[[u32; 16]; 8]; 2],
        next: [[[u32; 16]; 8]; 2],
        into: [[[u32; 16]; 8]; 2],
    }

    fn plan(tile_words: usize) -> &'static Plan {
        static MADE: std::sync::OnceLock<Vec<Option<Plan>>> = std::sync::OnceLock::new();
        MADE.get_or_init(|| {
            (0..=64)
                .map(|tw: usize| {
                    (tw % 16 == 0 && tw >= 16).then(|| {
                        let (nw, b) = (tw / 2, tw / 16);
                        let bits = nw * 32;
                        let mut p = Plan { first: [[[0; 16]; 8]; 2], next: [[[0; 16]; 8]; 2], into: [[[0; 16]; 8]; 2] };
                        for h in 0..2 {
                            for j in 0..8 {
                                for i in 0..16 {
                                    let l = 16 * h + i;
                                    let at = (b * (8 * l + j + 1) + bits - 16) % bits;
                                    p.first[h][j][i] = (at / 32) as u32;
                                    p.next[h][j][i] = ((at / 32 + 1) % nw) as u32;
                                    p.into[h][j][i] = (at % 32) as u32;
                                }
                            }
                        }
                        p
                    })
                })
                .collect()
        })[tile_words]
            .as_ref()
            .expect("a whole number of bits a weight")
    }

    /// Sixteen f16 values from `bytes` (little-endian) as f32s.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn f16s(bytes: *const u8) -> __m512 {
        _mm512_cvtph_ps(_mm256_loadu_si256(bytes as *const __m256i))
    }

    /// Sixteen values rounded to f16 (to nearest, ties to even) and back.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn round16(v: __m512) -> __m512 {
        _mm512_cvtph_ps(_mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v))
    }

    /// [`super::scaled_halves`] on the vectors.
    ///
    /// # Safety
    /// The CPU must have AVX-512 F ([`available`]).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn scaled_halves(x: &[f32], scales: &[u8], out: &mut [f32]) {
        let n = out.len();
        assert!(x.len() >= n && scales.len() >= 2 * n);
        let whole = n / 16 * 16;
        for i in (0..whole).step_by(16) {
            let v = round16(_mm512_loadu_ps(x.as_ptr().add(i)));
            _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_mul_ps(v, f16s(scales.as_ptr().add(2 * i))));
        }
        for i in whole..n {
            out[i] = super::half(x[i]) * half::f16::from_bits(u16::from_le_bytes([scales[2 * i], scales[2 * i + 1]])).to_f32();
        }
    }

    /// [`super::scaled_by`] on the vectors.
    ///
    /// # Safety
    /// The CPU must have AVX-512 F ([`available`]).
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn scaled_by(v: &mut [f32], by: f32, scales: Option<&[u8]>) {
        let n = v.len();
        assert!(scales.is_none_or(|s| s.len() >= 2 * n));
        let (whole, b) = (n / 16 * 16, _mm512_set1_ps(by));
        for i in (0..whole).step_by(16) {
            let mut x = _mm512_mul_ps(_mm512_loadu_ps(v.as_ptr().add(i)), b);
            if let Some(s) = scales {
                x = _mm512_mul_ps(x, f16s(s.as_ptr().add(2 * i)));
            }
            _mm512_storeu_ps(v.as_mut_ptr().add(i), round16(x));
        }
        for i in whole..n {
            v[i] = match scales {
                Some(s) => super::half(v[i] * by * half::f16::from_bits(u16::from_le_bytes([s[2 * i], s[2 * i + 1]])).to_f32()),
                None => super::half(v[i] * by),
            };
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn ld(v: &[u32; 16]) -> __m512i {
        // SAFETY: sixteen u32s, read unaligned.
        unsafe { _mm512_loadu_si512(v.as_ptr() as *const __m512i) }
    }

    /// [`super::split_sums`] on the vectors (see the module's notes): split `rows` of tile columns `cols`, the tiles
    /// taken as they lie (tile row by tile row), each column's warps' sums kept in memory between their tiles.
    /// `trellis`: the projection's words as a record lays them out (little-endian u32s).
    ///
    /// # Safety
    /// The CPU must have AVX-512 F, BW and VNNI ([`available`]).
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn split_sums(trellis: &[u8], k: usize, n: usize, tile_words: usize, xh: &[f32], rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Vec<f32> {
        let (kts, ntiles, nw) = (k / 16, n / 16, tile_words / 2);
        assert!(tile_words % 16 == 0 && (16..=64).contains(&tile_words) && cols.end <= ntiles && rows.end <= kts && xh.len() >= k && trellis.len() >= kts * ntiles * nw * 4);
        let p = plan(tile_words);
        // the code vectors' word indices and shifts, and the input's four patterns (rows 2 (l % 4) + 0, 1, 8 and 9)
        let zero = _mm512_setzero_si512();
        let (mut first, mut next, mut into, mut back) = ([[zero; 8]; 2], [[zero; 8]; 2], [[zero; 8]; 2], [[zero; 8]; 2]);
        for h in 0..2 {
            for j in 0..8 {
                first[h][j] = ld(&p.first[h][j]);
                next[h][j] = ld(&p.next[h][j]);
                into[h][j] = ld(&p.into[h][j]);
                back[h][j] = _mm512_sub_epi32(_mm512_set1_epi32(32), into[h][j]);
            }
        }
        let mut xidx = [zero; 4];
        for (m, o) in [0u32, 1, 8, 9].into_iter().enumerate() {
            let mut at = [0u32; 16];
            for (i, a) in at.iter_mut().enumerate() {
                *a = o + 2 * (i as u32 % 4);
            }
            xidx[m] = ld(&at);
        }
        // the tile's words past the first sixteen (a tile of 3 bits a weight has 24)
        let rest: __mmask16 = if nw >= 32 { 0xffff } else { ((1u32 << (nw - 16)) - 1) as __mmask16 };
        // each column's warps' four vectors of sums: lanes 0..16's lo and hi, lanes 16..32's (the thread's own, kept
        // from task to task: a task's 147 KB of them made anew was most of its time, 32 threads at the heap at once)
        thread_local! {
            static ACC: std::cell::RefCell<Vec<[[f32; 64]; 8]>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        ACC.with_borrow_mut(|acc| {
            acc.clear();
            acc.resize(cols.len(), [[0f32; 64]; 8]);
            sums_into(acc, trellis, ntiles, nw, xh, rows, cols, &first, &next, &into, &back, &xidx, rest)
        })
    }

    /// [`split_sums`]' tiles into `acc` and the columns' sums of them.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn sums_into(
        acc: &mut [[[f32; 64]; 8]],
        trellis: &[u8],
        ntiles: usize,
        nw: usize,
        xh: &[f32],
        rows: std::ops::Range<usize>,
        cols: std::ops::Range<usize>,
        first: &[[__m512i; 8]; 2],
        next: &[[__m512i; 8]; 2],
        into: &[[__m512i; 8]; 2],
        back: &[[__m512i; 8]; 2],
        xidx: &[__m512i; 4],
        rest: __mmask16,
    ) -> Vec<f32> {
        let (mul, ones, base) = (_mm512_set1_epi32(0x83dc_d12du32 as i32), _mm512_set1_epi32(0x0101_0101), _mm512_set1_epi32(1024));
        let (scale, shift) = (_mm512_set1_ps(1774.0 / 262_144.0), _mm512_set1_ps(10.382_812_5));
        for kt in rows.clone() {
            let w = (kt - rows.start) % 8;
            let x = _mm512_loadu_ps(xh.as_ptr().add(kt * 16));
            let xv = [_mm512_permutexvar_ps(xidx[0], x), _mm512_permutexvar_ps(xidx[1], x), _mm512_permutexvar_ps(xidx[2], x), _mm512_permutexvar_ps(xidx[3], x)];
            for nt in cols.clone() {
                let at = trellis.as_ptr().add((kt * ntiles + nt) * nw * 4);
                // (the tile's nw words lie within `trellis`, asserted above; the second load reads only the words past
                // sixteen, masked)
                let t0 = _mm512_loadu_si512(at as *const __m512i);
                let t1 = _mm512_maskz_loadu_epi32(rest, at.add(64) as *const i32);
                let sums = acc[nt - cols.start][w].as_mut_ptr();
                let mut a = [_mm512_loadu_ps(sums), _mm512_loadu_ps(sums.add(16)), _mm512_loadu_ps(sums.add(32)), _mm512_loadu_ps(sums.add(48))];
                for h in 0..2 {
                    for j in 0..8 {
                        let wa = _mm512_permutex2var_epi32(t0, first[h][j], t1);
                        let wb = _mm512_permutex2var_epi32(t0, next[h][j], t1);
                        let code = _mm512_srli_epi32::<16>(_mm512_or_si512(_mm512_sllv_epi32(wa, into[h][j]), _mm512_srlv_epi32(wb, back[h][j])));
                        let sum = _mm512_dpbusd_epi32(base, _mm512_mullo_epi32(code, mul), ones);
                        let v = _mm512_fmsub_ps(_mm512_cvtepi32_ps(sum), scale, shift);
                        let w16 = _mm512_cvtph_ps(_mm512_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v));
                        let i = 2 * h + j / 4;
                        a[i] = _mm512_fmadd_ps(xv[j % 4], w16, a[i]);
                    }
                }
                _mm512_storeu_ps(sums, a[0]);
                _mm512_storeu_ps(sums.add(16), a[1]);
                _mm512_storeu_ps(sums.add(32), a[2]);
                _mm512_storeu_ps(sums.add(48), a[3]);
            }
        }
        // each column's sums as `super::column_sums` adds them: lane l's lo (columns 0..8) or hi, the warps in order
        let mut out = vec![0f32; cols.len() * 16];
        for (col, sums) in acc.iter().zip(out.chunks_exact_mut(16)) {
            for (c, sum) in sums.iter_mut().enumerate() {
                let mut total = 0.0f32;
                for warp in col {
                    for q in 0..4 {
                        let l = (c % 8) * 4 + q;
                        total += warp[(2 * (l / 16) + c / 8) * 16 + l % 16];
                    }
                }
                *sum = total;
            }
        }
        out
    }
}

/// One decode step's expert of `record` (laid out as `layout` says) on the host's thread, as [`Exl3Slots`] makes a
/// job of one row on a card: `[hidden]`, unweighted. `table`: [`silu_table`]'s for `limit`. A test's: a step's
/// experts go over the host's cores together ([`step_experts_host`]).
pub fn step_expert_host(record: &[u8], layout: &RecordLayout, x: &[f32], limit: f32, table: &[f32], fused: bool) -> Vec<f32> {
    let [g, u, d] = layout.proj;
    let project = |p: &ProjAt, xh: &[f32]| -> Vec<f32> {
        let sums = p.step_sums(record, xh, 0..p.n / 16, fused);
        p.finish(record, sums, 1)
    };
    let (gate, up) = (project(&g, &g.prepare(record, x, 1)), project(&u, &u.prepare(record, x, 1)));
    project(&d, &d.prepare_swiglu(record, &gate, &up, limit, table))
}

/// Tile columns a task of [`step_experts_host`] takes (one 128-output block, every split of it): a gate's or an up's
/// 144 in 18 tasks, a down's 320 in 40, each its block's whole output (its splits' sums added in order and the output
/// transform made, a block's Hadamard transform its own).
const STEP_TASK_COLS: usize = 8;

/// A decode step's experts (`(record, x)` each, laid out as `layout` says) on the host's cores, each as
/// [`step_expert_host`] makes it, to the bit: the gates' and ups' inputs, then their outputs in tasks of an output
/// block of an expert's each, the SwiGLU, then every down's alike. Each output `[hidden]`, unweighted. On the workers
/// that stay ([`team::team`]), which keep the cores awake between a step's layers. `fused`: as [`ProjAt::step_sums`].
pub fn step_experts_host(jobs: &[(&[u8], &[f32])], layout: &RecordLayout, limit: f32, fused: bool) -> Vec<Vec<f32>> {
    let team = team::team();
    let table = silu_table(limit);
    let [g, u, d] = layout.proj;
    // `f(i)` for each `i` below `n`, on the team
    let each = |n: usize, f: &(dyn Fn(usize) -> Vec<f32> + Sync)| -> Vec<Vec<f32>> {
        let out: Vec<std::sync::OnceLock<Vec<f32>>> = (0..n).map(|_| std::sync::OnceLock::new()).collect();
        team.run(n, &|i| {
            let _ = out[i].set(f(i));
        });
        out.into_iter().map(|o| o.into_inner().expect("every task ran")).collect()
    };
    // each `(projection, job)`'s output: tasks of a split's tile rows over a part of the columns (a split's rows lie
    // together in the record, so a task reads runs of a tile row's columns as they lie, where a task of one block's
    // columns down every row read 768 bytes of each row's 14 KB, the team's eighteen at once: half the rate one thread
    // reads them in order at), then each block's splits' sums added in order and its output transform made
    let pass = |which: &[(&ProjAt, &[Vec<f32>])]| -> Vec<Vec<Vec<f32>>> {
        let splits: Vec<usize> = which.iter().map(|(p, _)| step_splits(p.k, p.n)).collect();
        let units = splits.iter().sum::<usize>() * jobs.len();
        // (each projection's columns in parts of whole blocks: enough tasks for the team to share, a few each)
        let spans: Vec<Vec<std::ops::Range<usize>>> = which
            .iter()
            .map(|(p, _)| {
                let (cols, blocks) = (p.n / 16, (p.n / 16).div_ceil(STEP_TASK_COLS));
                let per = blocks.div_ceil((3 * team.size()).div_ceil(units.max(1)).clamp(1, blocks)) * STEP_TASK_COLS;
                (0..cols).step_by(per).map(|c| c..(c + per).min(cols)).collect()
            })
            .collect();
        let (mut tasks, mut base) = (Vec::new(), Vec::new());
        for w in 0..which.len() {
            base.push(tasks.len());
            for j in 0..jobs.len() {
                for s in 0..splits[w] {
                    for c in 0..spans[w].len() {
                        tasks.push((w, j, s, c));
                    }
                }
            }
        }
        let partials = each(tasks.len(), &|i| {
            let (w, j, s, c) = tasks[i];
            let (p, xs) = which[w];
            p.split_sums(jobs[j].0, &xs[j], s, spans[w][c].clone(), fused)
        });
        let mut blocks = Vec::new();
        for (w, (p, _)) in which.iter().enumerate() {
            for j in 0..jobs.len() {
                for b in (0..p.n / 16).step_by(STEP_TASK_COLS) {
                    blocks.push((w, j, b));
                }
            }
        }
        let made = each(blocks.len(), &|i| {
            let (w, j, b) = blocks[i];
            let p = which[w].0;
            let cols = b..(b + STEP_TASK_COLS).min(p.n / 16);
            let c = spans[w].iter().position(|r| r.contains(&b)).expect("a part holds every block");
            let at = (b - spans[w][c].start) * 16;
            // (as `ProjAt::step_sums` adds them: from 0, the splits in order)
            let mut sums = vec![0f32; cols.len() * 16];
            for s in 0..splits[w] {
                let part = &partials[base[w] + (j * splits[w] + s) * spans[w].len() + c][at..at + sums.len()];
                for (a, v) in sums.iter_mut().zip(part) {
                    *a += v;
                }
            }
            p.finish_cols(jobs[j].0, sums, cols)
        });
        let mut out: Vec<Vec<Vec<f32>>> = which.iter().map(|(p, _)| vec![vec![0f32; p.n]; jobs.len()]).collect();
        for (&(w, j, b), y) in blocks.iter().zip(&made) {
            out[w][j][b * 16..b * 16 + y.len()].copy_from_slice(y);
        }
        out
    };
    let mut xs = each(2 * jobs.len(), &|i| [&g, &u][i % 2].prepare_row(jobs[i / 2].0, jobs[i / 2].1)).into_iter();
    let (mut xg, mut xu) = (Vec::with_capacity(jobs.len()), Vec::with_capacity(jobs.len()));
    while let (Some(a), Some(b)) = (xs.next(), xs.next()) {
        xg.push(a);
        xu.push(b);
    }
    let mut gu = pass(&[(&g, &xg), (&u, &xu)]).into_iter();
    let (gate, up) = (gu.next().expect("the gates"), gu.next().expect("the ups"));
    let xd = each(jobs.len(), &|j| d.prepare_swiglu(jobs[j].0, &gate[j], &up[j], limit, &table));
    pass(&[(&d, &xd)]).pop().expect("the downs")
}

impl ProjAt {
    /// [`Self::finish`] of tile columns `cols` alone (whole 128-output blocks: a block's transform is its own), from
    /// their sums (`[16 · cols]`, one row): rounded, transformed and scaled by `svh`, as the card's output transform
    /// makes them; the scales read from `record` for those columns alone.
    pub fn finish_cols(&self, record: &[u8], mut y: Vec<f32>, cols: std::ops::Range<usize>) -> Vec<f32> {
        assert!(cols.start % 8 == 0 && cols.len() % 8 == 0 && y.len() == cols.len() * 16, "whole blocks of 128 outputs");
        // (each rounding sixteen at a time where the CPU has AVX-512: `scaled_by`)
        scaled_by(&mut y, 1.0, None);
        had(&mut y);
        let at = self.svh + 2 * cols.start * 16;
        scaled_by(&mut y, ISQRT128, Some(&record[at..at + 2 * cols.len() * 16]));
        y
    }
}
