//! A decode step's dense calls: the buffers and bind groups a device keeps for them, and a call's projections,
//! units and chains in one submit, begun and read back.

use super::kernels::{one_name, one_shader, round_stage, swiglu_stage};
use super::*;

/// OAIY_DENSE_FEW: one row of x through [`FEW_KERNEL`] and a call's own buffers, as two to eight rows are (to compare
/// [`Arena`]'s way with).
fn few_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_DENSE_FEW").is_some())
}

/// What a decode step's dense calls share on a device, made at the first and kept: the call's inputs one after another
/// in one buffer (one write), its dispatches' parameters in another (one write, each dispatch's at an offset its bind
/// group takes), its results in a third and their read-back, and each weight buffer's bind group with those. A call
/// made its inputs' buffers, a result and a parameter buffer and a bind group for every weight, and a read-back, wrote
/// each on its own, and let them all go after: in a reply that was 300 us a call beyond the GPU's 50 (a quarter of a
/// decode step's time over its 380 calls), where the same call alone in a test took 80.
pub(crate) struct Arena {
    pipeline_layout: wgpu::PipelineLayout,
    layout: wgpu::BindGroupLayout,
    x: wgpu::Buffer,
    y: wgpu::Buffer,
    staging: wgpu::Buffer,
    params: wgpu::Buffer,
    /// bytes from one dispatch's parameters to the next's (the device's alignment of a uniform's offset)
    step: u32,
    groups: std::collections::HashMap<wgpu::Buffer, wgpu::BindGroup>,
    /// The stages' bind group ([`STAGE`]): the results read, the inputs written.
    stage: Option<wgpu::BindGroup>,
}

/// The most a call's inputs, and its results, may hold to go through the [`Arena`] (bytes), and its most dispatches.
const ARENA_BYTES: u64 = 4 << 20;
const ARENA_DISPATCHES: usize = 256;
/// A dispatch's parameters (the kernels' `p`).
const PARAM_BYTES: u64 = 48;

/// Bytes from one dispatch's parameters to the next's: the least multiple of the device's alignment for a uniform's
/// offset that holds them (64 on an NVIDIA card, 256 on a Mac and through Direct3D; the larger of the two alone
/// gave 48 where the alignment is 32, which is no multiple of it).
fn param_step(alignment: u32) -> u32 {
    (PARAM_BYTES as u32).next_multiple_of(alignment.max(16))
}

#[cfg(test)]
#[test]
fn a_dispatchs_parameters_start_at_a_multiple_of_the_devices_alignment() {
    for (alignment, step) in [(16u32, 48u32), (32, 64), (64, 64), (256, 256)] {
        assert_eq!(param_step(alignment), step, "alignment {alignment}");
        assert_eq!(step % alignment, 0);
    }
}

impl Arena {
    fn new(gpu: &Gpu) -> Arena {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("oaiy-dense-arena"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: std::num::NonZeroU64::new(PARAM_BYTES) }),
            ],
        });
        let pipeline_layout =
            gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("oaiy-dense-arena"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let step = param_step(gpu.limits.min_uniform_buffer_offset_alignment);
        let buffer = |label, size, usage| gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        Arena {
            pipeline_layout,
            layout,
            x: buffer("oaiy-dense-arena-x", ARENA_BYTES, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            y: buffer("oaiy-dense-arena-y", ARENA_BYTES, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            staging: buffer("oaiy-dense-arena-read", ARENA_BYTES, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            params: buffer("oaiy-dense-arena-params", ARENA_DISPATCHES as u64 * step as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            step,
            groups: std::collections::HashMap::new(),
            stage: None,
        }
    }

    /// The stages' bind group, made the first time it is asked for.
    fn stage(&mut self, gpu: &Gpu) -> &wgpu::BindGroup {
        let Arena { layout, x, y, params, stage, .. } = self;
        stage.get_or_insert_with(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-dense-arena-stage"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gpu.dummy().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: params, offset: 0, size: std::num::NonZeroU64::new(PARAM_BYTES) }) },
                ],
            })
        })
    }

    /// `weights`' bind group, made the first time it is asked for.
    fn group(&mut self, gpu: &Gpu, weights: &wgpu::Buffer) -> &wgpu::BindGroup {
        let Arena { layout, x, y, params, groups, .. } = self;
        groups.entry(weights.clone()).or_insert_with(|| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-dense-arena"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: weights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: params, offset: 0, size: std::num::NonZeroU64::new(PARAM_BYTES) }) },
                ],
            })
        })
    }
}

/// The arena's bind groups of `buffers` let go (with the weights they are of: a kept group keeps its buffers).
pub(super) fn forget(gpu: &Gpu, buffers: &mut dyn Iterator<Item = &wgpu::Buffer>) {
    if let Some(arena) = gpu.dense_arena.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        for buffer in buffers {
            arena.groups.remove(buffer);
        }
    }
}

/// An expert of a decode step for [`forward_units`]: its gate and up projections of `x` (one row, as they take it),
/// and the down projection of their SwiGLU times `weight` (1 where it has none).
pub struct Unit<'a> {
    pub gate: &'a DenseGpu,
    pub up: &'a DenseGpu,
    pub down: &'a DenseGpu,
    pub x: &'a [f32],
    pub weight: f32,
}

/// A projection of other projections' results for [`forward_chained`]: `first`'s sums of one row each (`(weight, x,
/// rows)`), rounded to bf16 and laid end to end, are the row `then` multiplies, quantized to fp8 first where
/// `quantized` (as an fp8 weight takes its activation).
pub struct Chain<'a> {
    pub first: &'a [(&'a DenseGpu, &'a [f32], Range<usize>)],
    pub then: &'a DenseGpu,
    pub quantized: bool,
}

/// A decode step's experts on one device in one submit and one read back: each unit's gate and up projections of its
/// row, their SwiGLU quantized on the device ([`swiglu_stage`]) and its down projection (its sums, f32), and with them
/// the sums of `others` (plain weights against one row each, `(weight, x, rows)`). None when it does not go through
/// the device's [`Arena`] (past its sizes, or OAIY_DENSE_FEW): the caller then makes the projections in calls of their
/// own with the SwiGLU on the host between them, which was two round trips an expert's step and is what a prompt's
/// rows still take.
pub fn forward_units(units: &[Unit<'_>], others: &[(&DenseGpu, &[f32], Range<usize>)], limit: f32) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let (downs, _, sums) = begin_units(units, others, limit)?.finish();
    Some((downs, sums))
}

/// [`forward_units`] begun: the call submitted, its results [`Pending`] until they are asked for. What the caller
/// does meanwhile runs beside the device's work: a decode step's experts on the host's cores beside a card's, with no
/// thread started for the card.
pub fn begin_units(units: &[Unit<'_>], others: &[(&DenseGpu, &[f32], Range<usize>)], limit: f32) -> Option<Pending> {
    let first = units.first().map(|u| u.gate).or(others.first().map(|o| o.0))?;
    let gpu = &first.gpu;
    assert!(units.iter().flat_map(|u| [u.gate, u.up, u.down]).chain(others.iter().map(|o| o.0)).all(|w| Arc::ptr_eq(&w.gpu, gpu)), "dense: a call on more than one GPU");
    let _one = first.serial.lock().unwrap_or_else(|p| p.into_inner());
    arena_begin(gpu, others, units, &[], limit)
}

/// A chain's last projection's sums (f32), all of it in one submit and one read back: the first projections, their
/// row rounded and quantized on the device ([`round_stage`]), and the last. Two calls with the rounding on the host
/// between them give the same to the bit. None when it does not go through the device's [`Arena`] (the row not the
/// last's width or not whole 32s, past the arena's sizes, or OAIY_DENSE_FEW): the caller then makes those two calls.
pub fn forward_chained(chain: &Chain<'_>) -> Option<Vec<f32>> {
    let gpu = &chain.then.gpu;
    assert!(chain.first.iter().all(|(w, ..)| Arc::ptr_eq(&w.gpu, gpu)), "dense: a call on more than one GPU");
    let _one = chain.then.serial.lock().unwrap_or_else(|p| p.into_inner());
    arena_begin(gpu, &[], &[], std::slice::from_ref(chain), 0.0)?.finish().1.pop()
}

/// [`forward_batch`] for a call whose every item is one row of x, through the device's [`Arena`]: None when it is not
/// such a call or is past the arena's sizes (the caller then makes its own buffers).
pub(super) fn forward_in_arena(gpu: &Arc<Gpu>, items: &[(&DenseGpu, &[f32], usize, Range<usize>)]) -> Option<Vec<Vec<f32>>> {
    if items.iter().any(|(_, _, t, rows)| *t != 1 || rows.is_empty()) {
        return None;
    }
    let plain: Vec<(&DenseGpu, &[f32], Range<usize>)> = items.iter().map(|(w, x, _, rows)| (*w, *x, rows.clone())).collect();
    arena_begin(gpu, &plain, &[], &[], 0.0).map(|pending| pending.finish().2)
}

/// A dispatch of an [`arena_begin`]: the weight's buffer it reads, its kernel's kind and grid, and its parameters.
struct Step<'a> {
    buffer: &'a wgpu::Buffer,
    kind: Kind,
    grid: u32,
    params: [u32; 12],
}

/// `w`'s dispatches for rows `rows` of it against the row at `xo` of the inputs, its sums to `yo` of the results.
fn steps_of<'a>(w: &'a DenseGpu, rows: &Range<usize>, xo: u32, yo: u32, steps: &mut Vec<Step<'a>>) {
    assert!(rows.end <= w.n, "dense: rows past the weight");
    for (buffer, first, n_rows, soff, woff) in &w.chunks {
        let (a, b) = ((*first as usize).max(rows.start), (*first as usize + *n_rows as usize).min(rows.end));
        if a < b {
            steps.push(Step {
                buffer,
                kind: w.kind,
                grid: (b - a).div_ceil(8) as u32,
                params: [w.k as u32, 1, *first, *n_rows, rows.start as u32, rows.len() as u32, *soff, w.kind as u32, *woff, xo, yo, 0],
            });
        }
    }
}

/// An [`Arena`]'s call submitted and not yet read back ([`begin_units`]). The arena is the call's own until then:
/// another call on the device meanwhile makes itself one.
pub struct Pending {
    gpu: Arc<Gpu>,
    arena: Arena,
    /// the bytes to read back
    bytes: u64,
    /// how many sums each result holds: the units' down projections', the chains' last projections', the plain weights'
    lens: [Vec<usize>; 3],
}

impl Pending {
    /// The call's results, waited for: each unit's down sums, each chain's last projection's, the plain weights'.
    pub fn finish(self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let Pending { gpu, arena, bytes, lens } = self;
        let waiting = std::time::Instant::now();
        let raw = gpu.map_read_soon(&arena.staging, bytes);
        crate::profile::add(&crate::profile::DENSE_WAIT, waiting);
        let mut at = 0usize;
        let mut take = |n: &usize| -> Vec<f32> {
            let v = raw[at..at + n * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            at += n * 4;
            v
        };
        let [downs, thens, sums] = lens.map(|of| of.iter().map(&mut take).collect::<Vec<Vec<f32>>>());
        // the arena back for the next call (let go if one came back before it)
        gpu.dense_arena.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert(arena);
        (downs, thens, sums)
    }
}

/// One submit through the device's [`Arena`], left [`Pending`]: `plain` weights against one row each, `units`
/// ([`forward_units`]) and `chains` ([`forward_chained`]). The inputs go one after another (each once, however many
/// weights take it), then each unit's activation and each chain's row, which the stages write; the results are each
/// unit's gate and up sums and each chain's first sums, then what is read back: the units' down sums, the chains'
/// last projections' and `plain`'s. Three passes: the gate, up, first and plain projections; the stages; the down and
/// last projections. None when it does not go through an arena.
fn arena_begin(gpu: &Arc<Gpu>, plain: &[(&DenseGpu, &[f32], Range<usize>)], units: &[Unit<'_>], chains: &[Chain<'_>], limit: f32) -> Option<Pending> {
    if few_only() || plain.iter().any(|(_, _, rows)| rows.is_empty()) || chains.iter().any(|c| c.first.iter().any(|(_, _, rows)| rows.is_empty())) {
        return None;
    }
    let making = std::time::Instant::now();
    let mut inputs: Vec<((*const f32, usize), u32)> = Vec::new();
    let mut xs: Vec<u8> = Vec::new();
    let mut place = |x: &[f32], k: usize| -> u32 {
        assert_eq!(x.len(), k, "dense: the input is not [1, k]");
        if let Some((_, at)) = inputs.iter().find(|(key, _)| *key == (x.as_ptr(), x.len())) {
            return *at;
        }
        let at = (xs.len() / 4) as u32;
        inputs.push(((x.as_ptr(), x.len()), at));
        xs.extend(x.iter().flat_map(|v| v.to_le_bytes()));
        at
    };
    let unit_inputs: Vec<u32> = units
        .iter()
        .map(|u| {
            assert!(u.gate.k == u.up.k && u.gate.n == u.up.n && u.gate.n == u.down.k && u.gate.n % 32 == 0, "dense: a unit's projections do not fit each other");
            place(u.x, u.gate.k)
        })
        .collect();
    let chain_inputs: Vec<Vec<u32>> = chains.iter().map(|c| c.first.iter().map(|(w, x, _)| place(x, w.k)).collect()).collect();
    let plain_inputs: Vec<u32> = plain.iter().map(|(w, x, _)| place(x, w.k)).collect();
    let given = (xs.len() / 4) as u32;
    // the inputs' buffer after what is given: each unit's activation and each chain's row; the results: each unit's
    // gate and up sums and each chain's first sums, then (read back) each unit's down sums, each chain's last
    // projection's and the plain weights' sums
    let width: u32 = units.iter().map(|u| u.gate.n as u32).sum();
    let widths: Vec<u32> = chains.iter().map(|c| c.first.iter().map(|(_, _, rows)| rows.len() as u32).sum()).collect();
    if chains.iter().zip(&widths).any(|(c, &w)| w as usize != c.then.k || w % 32 != 0) {
        return None;
    }
    let kept = 2 * width + widths.iter().sum::<u32>();
    // (a stage: whether it is a unit's SwiGLU, else a chain's rounding, and its parameters)
    let (mut first, mut stage, mut last): (Vec<Step<'_>>, Vec<(bool, [u32; 12])>, Vec<Step<'_>>) = (Vec::new(), Vec::new(), Vec::new());
    let (mut act, mut sums, mut out) = (given, 0u32, kept);
    for (u, &xo) in units.iter().zip(&unit_inputs) {
        let inter = u.gate.n as u32;
        steps_of(u.gate, &(0..u.gate.n), xo, sums, &mut first);
        steps_of(u.up, &(0..u.up.n), xo, sums + inter, &mut first);
        stage.push((true, [sums, sums + inter, act, inter, u.weight.to_bits(), limit.to_bits(), 0, 0, 0, 0, 0, 0]));
        steps_of(u.down, &(0..u.down.n), act, out, &mut last);
        act += inter;
        sums += 2 * inter;
        out += u.down.n as u32;
    }
    for ((c, ins), &row) in chains.iter().zip(&chain_inputs).zip(&widths) {
        let from = sums;
        for ((w, _, rows), &xo) in c.first.iter().zip(ins) {
            steps_of(w, rows, xo, sums, &mut first);
            sums += rows.len() as u32;
        }
        stage.push((false, [from, 0, act, row, c.quantized as u32, 0, 0, 0, 0, 0, 0, 0]));
        steps_of(c.then, &(0..c.then.n), act, out, &mut last);
        act += row;
        out += c.then.n as u32;
    }
    for ((w, _, rows), &xo) in plain.iter().zip(&plain_inputs) {
        steps_of(w, rows, xo, out, &mut first);
        out += rows.len() as u32;
    }
    let dispatches = first.len() + stage.len() + last.len();
    if act as u64 * 4 > ARENA_BYTES || out as u64 * 4 > ARENA_BYTES || dispatches > ARENA_DISPATCHES || first.is_empty() {
        return None;
    }
    // (the device's arena taken for this call, or one made: it comes back when the call is read)
    let mut arena = gpu.dense_arena.lock().unwrap_or_else(|p| p.into_inner()).take().unwrap_or_else(|| Arena::new(gpu));
    let step = arena.step as usize;
    let mut table = vec![0u8; dispatches * step];
    for (i, p) in first.iter().map(|s| &s.params).chain(stage.iter().map(|(_, p)| p)).chain(last.iter().map(|s| &s.params)).enumerate() {
        for (j, v) in p.iter().enumerate() {
            table[i * step + 4 * j..i * step + 4 * j + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    gpu.queue().write_buffer(&arena.x, 0, &xs);
    gpu.queue().write_buffer(&arena.params, 0, &table);
    // (the pipelines and the groups first: a pass borrows them)
    let pipelines: Vec<Arc<wgpu::ComputePipeline>> =
        first.iter().chain(&last).map(|s| gpu.named_pipeline_in(one_name(s.kind), &arena.pipeline_layout, || one_shader(s.kind))).collect();
    let swiglu = stage.iter().any(|(unit, _)| *unit).then(|| gpu.named_pipeline_in("dense-swiglu", &arena.pipeline_layout, swiglu_stage));
    let round = stage.iter().any(|(unit, _)| !*unit).then(|| gpu.named_pipeline_in("dense-round", &arena.pipeline_layout, round_stage));
    for s in first.iter().chain(&last) {
        arena.group(gpu, s.buffer);
    }
    if !stage.is_empty() {
        arena.stage(gpu);
    }
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let project = |enc: &mut wgpu::CommandEncoder, steps: &[Step<'_>], pipelines: &[Arc<wgpu::ComputePipeline>], at: usize| {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (i, (s, pipeline)) in steps.iter().zip(pipelines).enumerate() {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &arena.groups[s.buffer], &[((at + i) * step) as u32]);
                pass.dispatch_workgroups(s.grid, 1, 1);
            }
        };
        project(&mut enc, &first, &pipelines[..first.len()], 0);
        if !stage.is_empty() {
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (i, (unit, p)) in stage.iter().enumerate() {
                    let pipeline = match unit {
                        true => swiglu.as_ref(),
                        false => round.as_ref(),
                    };
                    pass.set_pipeline(pipeline.expect("made above"));
                    pass.set_bind_group(0, arena.stage.as_ref().expect("made above"), &[((first.len() + i) * step) as u32]);
                    pass.dispatch_workgroups(p[3] / 32, 1, 1);
                }
            }
            project(&mut enc, &last, &pipelines[first.len()..], first.len() + stage.len());
        }
    }
    let bytes = (out - kept) as u64 * 4;
    enc.copy_buffer_to_buffer(&arena.y, kept as u64 * 4, &arena.staging, 0, bytes);
    crate::profile::add(&crate::profile::DENSE_MAKE, making);
    let read = units
        .iter()
        .flat_map(|u| [(u.gate, u.gate.n), (u.up, u.up.n), (u.down, u.down.n)])
        .chain(chains.iter().flat_map(|c| c.first.iter().map(|(w, _, rows)| (*w, rows.len())).chain([(c.then, c.then.n)])))
        .chain(plain.iter().map(|(w, _, rows)| (*w, rows.len())));
    for (w, rows) in read {
        crate::profile::DENSE_BYTES[0].fetch_add(w.nbytes_of(rows), Ordering::Relaxed);
        crate::profile::DENSE_BYTES[1].fetch_add(1, Ordering::Relaxed);
    }
    let submitting = std::time::Instant::now();
    gpu.queue().submit([enc.finish()]);
    crate::profile::add(&crate::profile::DENSE_SUBMIT, submitting);
    let lens = [units.iter().map(|u| u.down.n).collect(), chains.iter().map(|c| c.then.n).collect(), plain.iter().map(|(_, _, rows)| rows.len()).collect()];
    Some(Pending { gpu: Arc::clone(gpu), arena, bytes, lens })
}
