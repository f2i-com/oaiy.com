//! EXL3 projections of one shape and bitrate kept together on a GPU ([`Exl3Group`]): a stack's parts, DeepSeek-V4.1's
//! `wo_a` a slice for each of its eight groups of heads. A step's row through every part goes as one pass of the
//! group kernels the experts' slots run (their input transforms, matmul and output transforms, a job a part), three
//! dispatches where each part's own were three apiece, each part's output its own projection's to the bit (the same
//! kernels, splits and order).

use super::*;

/// See the module's notes ([`WgpuBackend::exl3_group`]).
pub struct Exl3Group {
    g: Group,
    count: usize,
    used: Arc<AtomicU64>,
    nbytes: u64,
}

impl std::fmt::Debug for Exl3Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exl3Group({} of {}x{} at {} words a tile)", self.count, self.g.n, self.g.k, self.g.tw)
    }
}

impl Drop for Exl3Group {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes, Ordering::Relaxed);
    }
}

impl Exl3Group {
    /// `(k, n)` of each part.
    pub fn kn(&self) -> (usize, usize) {
        (self.g.k, self.g.n)
    }

    /// How many parts.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl WgpuBackend {
    /// `parts` together on this GPU, counted in its weight budget: each of one shape and bitrate, its maps the identity,
    /// its words below a binding's limit together. None where they are not, or the budget has no room.
    pub fn exl3_group(&self, parts: &[&Exl3Data]) -> Option<Exl3Group> {
        let first = parts.first()?;
        let (k, n, tw) = (first.suh.len(), first.svh.len(), first.tile_words);
        let identity = |d: &Exl3Data| d.input_map.iter().enumerate().all(|(i, &m)| m as usize == i) && d.output_map.iter().enumerate().all(|(i, &m)| m as usize == i);
        if !parts.iter().all(|d| (d.suh.len(), d.svh.len(), d.tile_words) == (k, n, tw) && identity(d)) || k % 128 != 0 || n % 128 != 0 {
            return None;
        }
        let mw = k / 16 * n / 16 * (tw / 2);
        let nbytes = (parts.len() * (mw * 4 + (k + n) * 4)) as u64;
        if (parts.len() * mw * 4) as u64 > chunk_limit(&self.gpu.limits) {
            return None;
        }
        let prev = self.used.fetch_add(nbytes, Ordering::Relaxed);
        if prev + nbytes > self.budget {
            self.used.fetch_sub(nbytes, Ordering::Relaxed);
            return None;
        }
        let mut bytes = Vec::with_capacity(parts.len() * mw * 4);
        let (mut suh, mut svh) = (Vec::with_capacity(parts.len() * k), Vec::with_capacity(parts.len() * n));
        for d in parts {
            if d.words.len() != mw {
                self.used.fetch_sub(nbytes, Ordering::Relaxed);
                return None;
            }
            bytes.extend(d.words.iter().flat_map(|w| w.to_le_bytes()));
            suh.extend_from_slice(&d.suh);
            svh.extend_from_slice(&d.svh);
        }
        let words = self.gpu.upload_rows(&bytes, bytes.len(), 1).remove(0).0;
        let up = |v: &[f32]| {
            let d = self.vec(v.len());
            DeviceChain::upload(self, &d, v);
            d
        };
        let splits = 4096usize.div_ceil(n / 16).next_power_of_two().min(8).min(k / 16).max(1) as u32;
        let g = Group { words, suh: up(&suh), svh: up(&svh), k, n, tw, mwords: mw, splits };
        Some(Exl3Group { g, count: parts.len(), used: Arc::clone(&self.used), nbytes })
    }
}

/// What a pass of `jobs` of a group needs beside its input and output, taken (and its job list written) before anything
/// is recorded, as every vector of a recording is ([`run_program`]): the job list, the transformed rows, the partial
/// sums.
pub(super) struct GroupScratch {
    jobs: DeviceVec,
    count: usize,
    xh: DeviceVec,
    part: DeviceVec,
}

impl Exl3Group {
    /// [`GroupScratch`] for `jobs` (`(part, row of x)` each), on the device for good (a program's arena's: the job list
    /// written once).
    pub(super) fn scratch_kept(&self, b: &WgpuBackend, jobs: &[(u32, u32)]) -> GroupScratch {
        let list: Vec<f32> = jobs.iter().flat_map(|&(m, r)| [f32::from_bits(m), f32::from_bits(r)]).collect();
        let jv = b.vec(list.len());
        DeviceChain::upload(b, &jv, &list);
        let count = jobs.len();
        GroupScratch { jobs: jv, count, xh: b.vec(count * self.g.k), part: b.vec(count * self.g.splits as usize * self.g.n) }
    }

    /// The pass recorded into `rec`: each job's sums into row `j` of `y` (`[jobs, n]`), x's rows `k` wide.
    pub(super) fn record(&self, rec: &mut crate::chain::Recorder<'_>, s: &GroupScratch, x: &DeviceVec, y: &DeviceVec) {
        group_pass(rec, &self.g, x, false, &s.jobs, s.count, Order::Jobs, &s.xh, &s.part, y, 0.0, None);
    }
}
