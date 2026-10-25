//! A GGUF's matrices of one shape and type kept together on a GPU ([`QuantGroup`]), as [`crate::exl3::Exl3Group`]
//! keeps an EXL3 checkpoint's: a stack's parts, DeepSeek-V4.1's `wo_a` a slice for each of its eight groups of heads. A
//! step's row through every part goes as one pass of the experts' few-row kernel (a job a part), where each part's own
//! was its input's copy, its quantization, its matmul and its output's copy.

use super::*;

/// See the module's notes ([`WgpuBackend::quant_group`]).
pub struct QuantGroup {
    g: Group,
    count: usize,
    used: Arc<std::sync::atomic::AtomicU64>,
    nbytes: u64,
}

impl std::fmt::Debug for QuantGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantGroup({} of {}x{} in {})", self.count, self.g.n, self.g.k, self.g.kind.tag())
    }
}

impl Drop for QuantGroup {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes, Ordering::Relaxed);
    }
}

impl WgpuBackend {
    /// `parts` (each a matrix's blocks, `n` rows of `k` weights of `dtype`) together on this GPU, counted in its weight
    /// budget, their rows as the experts' kernels read them. None where the kernels do not decode the type, the rows are
    /// not whole blocks, the words pass a binding's limit, or the budget has no room.
    pub fn quant_group(&self, dtype: GgmlType, parts: &[&[u8]], n: usize, k: usize) -> Option<QuantGroup> {
        let kind = Kind::of(dtype)?;
        let (rb, rw) = (row_bytes(dtype, k), kind.row_words(k));
        if parts.is_empty() || !kind.fits(k) || parts.iter().any(|p| p.len() != n * rb) {
            return None;
        }
        let mwords = n * rw;
        let nbytes = (parts.len() * mwords * 4) as u64;
        if nbytes > chunk_limit(&self.gpu.limits) {
            return None;
        }
        let prev = self.used.fetch_add(nbytes, Ordering::Relaxed);
        if prev + nbytes > self.budget {
            self.used.fetch_sub(nbytes, Ordering::Relaxed);
            return None;
        }
        let mut words = vec![0u32; parts.len() * mwords];
        words.par_chunks_mut(rw).enumerate().for_each(|(r, row)| {
            let (m, r) = (r / n, r % n);
            kind.pack_row(&parts[m][r * rb..(r + 1) * rb], k, row);
        });
        let bytes: &[u8] = bytemuck::cast_slice(&words);
        let table = kind.table().map(|t| {
            let bytes: &[u8] = bytemuck::cast_slice(&t);
            self.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0
        });
        let g = Group { words: self.gpu.upload_rows(bytes, bytes.len(), 1).remove(0).0, kind, k, n, rw, mwords, table, cold: None, mats: 1 };
        Some(QuantGroup { g, count: parts.len(), used: Arc::clone(&self.used), nbytes })
    }
}

/// What a pass of `jobs` of a group needs beside its input and output: its job list, written once.
pub(crate) struct QuantGroupScratch {
    jobs: DeviceVec,
    count: usize,
}

impl QuantGroup {
    /// `(k, n)` of each part.
    pub fn kn(&self) -> (usize, usize) {
        (self.g.k, self.g.n)
    }

    /// How many parts.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Its parts' words' buffer (what knows it on its device).
    pub(crate) fn words(&self) -> &wgpu::Buffer {
        &self.g.words
    }

    /// [`QuantGroupScratch`] for `jobs` (`(part, row of x)` each), on the device for good (a program's arena's).
    pub(crate) fn scratch_kept(&self, b: &WgpuBackend, jobs: &[(u32, u32)]) -> QuantGroupScratch {
        let list: Vec<f32> = jobs.iter().flat_map(|&(m, r)| [f32::from_bits(m), f32::from_bits(r)]).collect();
        let jv = b.vec(list.len());
        DeviceChain::upload(b, &jv, &list);
        QuantGroupScratch { jobs: jv, count: jobs.len() }
    }

    /// The pass recorded into `rec`: each job's sums into row `j` of `y` (`[jobs, n]`), x's rows `k` wide.
    pub(crate) fn record(&self, rec: &mut crate::chain::Recorder<'_>, s: &QuantGroupScratch, x: &DeviceVec, y: &DeviceVec) {
        super::slots::pass(rec, &self.g, x, &s.jobs, s.count, Order::Jobs, y, self.count, false);
    }
}
