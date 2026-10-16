//! A card's cache of experts: the experts a run takes in (the admission kernel and the copy into their
//! slots), which are held, and what was counted.

use super::*;

/// An expert the card has no slot for, in a layer's state.
pub(super) const MISS: u32 = u32::MAX;

/// The experts a pass takes in at most a layer: the victims the host lists, and the copies a pass makes.
pub(super) const TAKE: usize = 32;

/// A pass's experts that the card has no slot for, seen to before the experts' kernels run. Each is given the slot of
/// one the host listed as least recently used (the state's victims, from `4 experts + 4`: [`TAKE`] experts, [`MISS`]
/// none; one this pass uses, or that has no slot any more, is passed over): the slots' map is changed here, and the
/// taking in listed for [`EXPERTS_IN`] (how many at `2 experts + 1`; each one's expert and slot after the victims).
/// When the victims are out, a prompt's (`p[0].z` 1) are numbered for the card's scratch (the state from `3 experts
/// + 4`: an expert's place, [`MISS`] none) and a few rows' are left for their kernels to read from the host's
/// memory. One workgroup: its threads mark the experts the pass's pairs name (`jobs`: a pair's expert its even
/// word) and say which have no slot; where any has none, one of them then goes through those pairs (a few rows':
/// the first 256 pairs') or the experts (a prompt's). Most layers of a step have none, and the one thread's loads are
/// one after another: going through every pair was 10 us a layer a pass. The experts not taken in are counted at
/// `2 experts + 2` (a prompt's numbered ones; a few rows' pairs left). `p[0]`: the pairs, the experts (1,024 at
/// most), whether the rest are numbered, the victims.
const ADMIT: &str = r#"
@group(0) @binding(0) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> state: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> mark: array<atomic<u32>, 1024>;
// the pairs whose expert has no slot: how many, and which of the first 256
var<workgroup> cold: atomic<u32>;
var<workgroup> none: array<u32, 256>;
// the victims tried, and the experts taken in
var<private> tried: u32;
var<private> taken: u32;

// expert `e` (no slot) given the next victim's that can go: whether there was one
fn take(e: u32) -> bool {
    let experts = p[0].y;
    let most = p[0].w;
    let victims = 4u * experts + 4u;
    while (tried < most) {
        let v = state[victims + tried];
        tried++;
        if (v < experts) {
            let slot = state[v];
            if (slot != 0xffffffffu && atomicLoad(&mark[v]) == 0u) {
                state[v] = 0xffffffffu;
                state[e] = slot;
                state[victims + most + 2u * taken] = e;
                state[victims + most + 2u * taken + 1u] = slot;
                taken++;
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let pairs = p[0].x;
    let experts = p[0].y;
    for (var e = t; e < experts; e += 256u) {
        atomicStore(&mark[e], 0u);
    }
    if (t == 0u) {
        atomicStore(&cold, 0u);
    }
    workgroupBarrier();
    for (var q = t; q < pairs; q += 256u) {
        let e = jobs[2u * q];
        atomicStore(&mark[e], 1u);
        let lacks = state[e] == 0xffffffffu;
        if (lacks) {
            atomicAdd(&cold, 1u);
        }
        if (q < 256u) {
            none[q] = u32(lacks);
        }
    }
    workgroupBarrier();
    if (t != 0u) {
        return;
    }
    tried = 0u;
    taken = 0u;
    var left = 0u;
    if (atomicLoad(&cold) != 0u) {
        if (p[0].z == 1u) {
            for (var e = 0u; e < experts; e++) {
                var at = 0xffffffffu;
                if (atomicLoad(&mark[e]) == 1u && state[e] == 0xffffffffu) {
                    if (!take(e)) {
                        at = left;
                        left++;
                    }
                }
                state[3u * experts + 4u + e] = at;
            }
        } else {
            for (var q = 0u; q < min(pairs, 256u); q++) {
                if (none[q] == 1u) {
                    let e = jobs[2u * q];
                    if (state[e] == 0xffffffffu) {
                        if (!take(e)) {
                            left++;
                        }
                    }
                }
            }
        }
    }
    state[2u * experts + 1u] = taken;
    state[2u * experts + 2u] = left;
}
"#;

/// Experts copied from the host's memory to the card: their gate and up matrices (`cg` to `hg`) and their down ones
/// (`cd` to `hd`), a workgroup 16,384 words, its 256 threads 256 consecutive words at each of 64 turns. A load from
/// the host's memory is the bus's fetch of its 64-byte line, kept for no later load: the threads that load at once
/// must share lines (each thread its own 64 words in turn had a warp's 32 loads in 32 lines, a line fetched for 4
/// bytes of it: 2.1 GB a second where the bus gives 26.6). `p[1].y` 1: the experts [`ADMIT`] took in, into their
/// slots (a workgroup's third index an entry of its list; past the list it ends at once); 0: the ones it numbered,
/// into the card's scratch at their places (the third index the expert; one with no place ends at once). `p[0]`: an
/// expert's words in the first group and in the second, the experts, the workgroups an expert's first group takes;
/// `p[1].x`: the victims ([`TAKE`]).
const EXPERTS_IN: &str = r#"
@group(0) @binding(0) var<storage, read> cg: array<u32>;
@group(0) @binding(1) var<storage, read> cd: array<u32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read_write> hg: array<u32>;
@group(0) @binding(7) var<storage, read_write> hd: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let experts = p[0].z;
    var e = wg.z;
    var at = 0xffffffffu;
    if (p[1].y == 1u) {
        if (wg.z < state[2u * experts + 1u]) {
            let entry = 4u * experts + 4u + p[1].x + 2u * wg.z;
            e = state[entry];
            at = state[entry + 1u];
        }
    } else {
        at = state[3u * experts + 4u + e];
    }
    if (at == 0xffffffffu) {
        return;
    }
    let gw = p[0].x;
    let dw = p[0].y;
    if (wg.x < p[0].w) {
        let first = wg.x * 16384u + t;
        for (var i = first; i < min(first + 16384u, gw); i += 256u) {
            hg[at * gw + i] = cg[e * gw + i];
        }
    } else {
        let first = (wg.x - p[0].w) * 16384u + t;
        for (var i = first; i < min(first + 16384u, dw); i += 256u) {
            hd[at * dw + i] = cd[e * dw + i];
        }
    }
}
"#;

/// A layer's routed experts where the card holds only some of them ([`QuantMoe::make`]'s `slots`): every expert's
/// matrices are in the host's memory ([`Gpu::host_buffer`]), which the card reads over the bus with no word from the
/// host between a layer's router and its experts. A pass takes the experts it uses that have no slot into the slots
/// of the least recently used before its experts' kernels run ([`ADMIT`], [`EXPERTS_IN`]: an expert crosses the bus
/// once, 57 us on eight lanes of PCIe 5), up to [`TAKE`] a layer; past that a few rows' kernels read an expert from
/// the host's memory where it is ([`cached_source`]) and a prompt's read it from the card's scratch, copied there.
/// The kernels write the pass each expert was last used in; a recording that ran the layer reads its state as it
/// finishes ([`Self::watch`]) and the host lists the next victims from it ([`Self::settle`]).
///
/// Flash-Next's GSQ-RCO IQ2_XS file is 37 GB of experts, 1.5 MB each, and one RTX 5090 holds 313 of a layer's 512.
/// Strata's request (4,086 tokens of code, then 256 of prose) there reads 24 experts a token from the host's
/// memory: whole layers' experts on the host's cores gave 46 tokens a second and a prompt of 24.5 s. A fixed set of
/// experts would read some 140 a token (the model's routing replayed: what a reply uses is not what its prompt did).
pub(crate) struct Cache {
    pub(super) gpu: Arc<Gpu>,
    pub(super) experts: usize,
    /// The experts the card holds.
    pub(super) slots: usize,
    /// Each expert's slot on the card ([`MISS`]: none); the step or check each was last used in; four words (the
    /// steps and checks so far, the experts the last pass took in and those it did not, the prompts' passes so far);
    /// the prompt's pass each was last used in; each expert's place in a prompt's scratch; the victims ([`TAKE`]
    /// experts); and the last pass's taking in (an expert and its slot each).
    pub(super) state: wgpu::Buffer,
    /// The gate and up group's, and the down group's: every expert's in the host's memory, the card's slots, and an
    /// expert's bytes there.
    pub(super) groups: [(wgpu::Buffer, wgpu::Buffer, u64); 2],
    pub(super) host: Mutex<Held>,
    /// The experts read from the host's memory (taken in or not), and those taken in, so far.
    pub(super) counts: [std::sync::atomic::AtomicU64; 2],
}

/// What the host knows of a part-held layer: the experts' slots when it last looked, the passes then (the steps' and
/// checks', and the prompts'), and the victims it listed.
pub(super) struct Held {
    pub(super) map: Vec<u32>,
    pub(super) seen: [u32; 2],
    pub(super) victims: Vec<u32>,
}

/// The most words a prompt's scratch of experts takes on a device (by its address): the largest of its part-held
/// layers' two groups', so a recording takes one pair of vectors for all of them ([`Cache::admit`]).
pub(super) static STAGE_MOST: Mutex<Vec<(usize, [usize; 2])>> = Mutex::new(Vec::new());

/// Every such layer's experts read from the host's memory and brought to a card since the process began.
static CACHED: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The experts cards have read from the host's memory, and those of them taken into a card's slots, since the process
/// began (every layer's that a card holds only some of): what a request's cost of them was is the difference over it.
pub fn cached_experts() -> (u64, u64) {
    (CACHED[0].load(Ordering::Relaxed), CACHED[1].load(Ordering::Relaxed))
}

impl Cache {
    /// Before a pass's experts' kernels (`jobs`: its `pairs`, a pair's expert its even word): its experts with no
    /// slot taken in, and (`staged`: a prompt's rows) those there was no victim for copied into the recording's
    /// scratch, where the tensor cores' kernels read them: the gate and up group's vector and the down group's
    /// (each layer's in turn the same two).
    pub(super) fn admit(&self, rec: &mut crate::chain::Recorder<'_>, jobs: &DeviceVec, pairs: usize, staged: bool) -> Option<[DeviceVec; 2]> {
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let (d, drw) = (rec.gpu().dummy().clone(), rec.gpu().dummy_rw().clone());
        // an expert's words in each group; a workgroup 16,384 of them
        let words = [self.groups[0].2 as usize / 4, self.groups[1].2 as usize / 4];
        let per = |w: usize| w.div_ceil(16384) as u32;
        let wide = per(words[0]) + per(words[1]);
        let params = |listed: u32| [words[0] as u32, words[1] as u32, self.experts as u32, per(words[0]), TAKE as u32, listed];
        rec.dispatch_wide("moe-admit", ADMIT, [&buf(jobs), &d, &d, &d, &d, &d, &self.state, &drw], &[pairs as u32, self.experts as u32, staged as u32, TAKE as u32], (1, 1, 1));
        rec.dispatch_wide(
            "moe-take-in",
            EXPERTS_IN,
            [&self.groups[0].0, &self.groups[1].0, &self.state, &d, &d, &d, &self.groups[0].1, &self.groups[1].1],
            &params(1),
            (wide, 1, TAKE.min(pairs) as u32),
        );
        if !staged {
            return None;
        }
        // room for every expert the card has no slot for: the recording's one pair of vectors, as long as the
        // device's largest such layer needs (a layer of another type a pair of its own would hold them all to the
        // recording's end)
        let lens = words.map(|w| (self.experts - self.slots) * w);
        let (held, stage) = match rec.moe_stage.take() {
            Some((l, s)) if l[0] >= lens[0] && l[1] >= lens[1] => (l, s),
            _ => {
                let device = Arc::as_ptr(&self.gpu) as usize;
                let most = STAGE_MOST.lock().unwrap_or_else(|p| p.into_inner()).iter().find(|(d, _)| *d == device).map_or(lens, |(_, m)| [m[0].max(lens[0]), m[1].max(lens[1])]);
                (most, most.map(|l| rec.scratch(l)))
            }
        };
        rec.moe_stage = Some((held, stage.clone()));
        rec.dispatch_wide(
            "moe-stage-copy",
            EXPERTS_IN,
            [&self.groups[0].0, &self.groups[1].0, &self.state, &d, &d, &d, &buf(&stage[0]), &buf(&stage[1])],
            &params(0),
            (wide, 1, self.experts as u32),
        );
        Some(stage)
    }

    /// `rec` reads the layer's state as it finishes (once a recording, however often it runs the layer).
    pub(super) fn watch(self: &Arc<Self>, rec: &mut crate::chain::Recorder<'_>) {
        if rec.settles.iter().any(|(c, _)| Arc::ptr_eq(c, self)) {
            return;
        }
        let at = rec.read_of(&self.state, 0, 3 * self.experts + 4);
        rec.settles.push((Arc::clone(self), at));
    }

    /// What a recording read of the layer's state once it had run (`read`: its first three parts): the card's
    /// experts that no pass since the last look used are listed as the next victims ([`TAKE`] of them), the ones
    /// steps and checks used longest ago first (then the ones prompts did), where that list is another than the card
    /// has. So a prompt's chunks take in what they use and leave each other's, and the slots they take are the ones
    /// a reply would miss least: a reply's experts are not its prompt's, and by one count of use a prompt put out
    /// the experts replies had just used before any other.
    pub(crate) fn settle(&self, read: &[f32]) {
        let e = self.experts;
        let word = |i: usize| read[i].to_bits();
        let mut host = self.host.lock().unwrap_or_else(|p| p.into_inner());
        let seen = host.seen;
        host.seen = [word(2 * e), word(2 * e + 3)];
        for x in 0..e {
            host.map[x] = word(x);
        }
        // (when a step or a check last used expert `x`, and when a prompt's rows did)
        let used = |x: usize| (word(e + x), word(2 * e + 4 + x));
        // (the layer's last pass: the experts it took in, and those it read where they are or from a prompt's scratch)
        let (taken, left) = (read[2 * e + 1].to_bits() as u64, read[2 * e + 2].to_bits() as u64);
        for (count, n) in [left + taken, taken].into_iter().enumerate() {
            self.counts[count].fetch_add(n, Ordering::Relaxed);
            CACHED[count].fetch_add(n, Ordering::Relaxed);
        }
        let mut victims: Vec<u32> = (0..e).filter(|&x| word(x) != MISS && used(x).0 <= seen[0] && used(x).1 <= seen[1]).map(|x| x as u32).collect();
        victims.sort_by_key(|&x| used(x as usize));
        victims.resize(TAKE, MISS);
        if victims != host.victims {
            self.gpu.write(&self.state, ((4 * e + 4) * 4) as u64, bytemuck::cast_slice(&victims[..]));
            host.victims = victims;
        }
    }
}
