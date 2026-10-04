//! VENDORED-LOCAL: resident EXL3 projections, original packed bytes.
use crate::CudaBackend;
use cudarc::driver::{
    result, sys, CudaGraph, CudaSlice, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use ggml_rs::{
    exl3::{Exl3Data, PackedLinear},
    Backend, Tensor,
};
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct Exl3Matrix {
    backend: Arc<CudaBackend>,
    words: CudaSlice<u32>,
    suh: CudaSlice<f32>,
    svh: CudaSlice<f32>,
    input: CudaSlice<u32>,
    output: CudaSlice<u32>,
    shape: [usize; 2],
    tile_words: i32,
    decode: Mutex<Option<DecodeScratch>>,
}
#[derive(Debug)]
struct DecodeScratch {
    graph: Option<ProjectionGraph>,
    splits: usize,
    input: CudaSlice<f32>,
    partial: CudaSlice<f32>,
}
impl Exl3Matrix {
    pub fn upload(backend: Arc<CudaBackend>, data: Exl3Data) -> Result<Self, String> {
        data.validate()?;
        let s = &backend.stream;
        let mut inverse = vec![0u32; data.output_map.len()];
        for (i, &v) in data.output_map.iter().enumerate() {
            inverse[v as usize] = i as u32;
        }
        let words = s.clone_htod(&data.words).map_err(|e| format!("{e:?}"))?;
        let suh = s.clone_htod(&data.suh).map_err(|e| format!("{e:?}"))?;
        let svh = s.clone_htod(&data.svh).map_err(|e| format!("{e:?}"))?;
        let input = s
            .clone_htod(&data.input_map)
            .map_err(|e| format!("{e:?}"))?;
        let output = s.clone_htod(&inverse).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            backend,
            words,
            suh,
            svh,
            input,
            output,
            shape: [data.svh.len(), data.suh.len()],
            tile_words: data.tile_words as i32,
            decode: Mutex::new(None),
        })
    }
}
impl PackedLinear for Exl3Matrix {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.words.len() * 4 + (self.suh.len() + self.svh.len()) * 8
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        self.linear_impl(x, false, None)
    }
}
impl Exl3Matrix {
    /// Scalar-layout CUDA reference for numerical and performance comparisons.
    pub fn linear_reference(&self, x: &Tensor) -> Tensor {
        self.linear_impl(x, true, None)
    }
    /// Explicit split count for CUDA tuning; automatic selection is used by `linear`.
    pub fn linear_with_splits(&self, x: &Tensor, splits: usize) -> Tensor {
        assert!(
            (1..=32).contains(&splits),
            "EXL3 splits must be between 1 and 32"
        );
        self.linear_impl(x, false, Some(splits))
    }
    fn decode_cached(&self, x: &Tensor, requested_splits: Option<usize>) -> Tensor {
        let b = &self.backend;
        let s = &b.stream;
        let (n, k) = (self.shape[0], self.shape[1]);
        let kernel = match self.tile_words {
            32 => "exl3_tile_32",
            48 => "exl3_tile_48",
            56 => "exl3_tile_56",
            64 => "exl3_tile_64",
            80 => "exl3_tile_80",
            96 => "exl3_tile_96",
            _ => "exl3_gemv_generic",
        };
        let tiled = kernel != "exl3_gemv_generic";
        let splits = if tiled {
            requested_splits.unwrap_or_else(|| {
                (4096usize.div_ceil(n / 16))
                    .next_power_of_two()
                    .min(8)
                    .min((k / 256).max(1))
            })
        } else {
            1
        };
        // Hold the lock until all three launches are queued. Reusing scratch is
        // safe across callers because every launch uses this same ordered stream.
        // Returned results have independent storage and are never recycled here.
        let mut cache = self.decode.lock().unwrap_or_else(|e| e.into_inner());
        if cache.as_ref().is_none_or(|c| c.splits != splits) {
            // Scratch kept from one step to the next cannot come from a captured step's arena.
            assert!(!s.graph_recording(), "EXL3 decode scratch made while a step is captured: decode once uncaptured first");
            // SAFETY: transforms/GEMV fully write these private buffers before
            // reading them; they outlive every launch and drop on this stream.
            *cache = Some(unsafe {
                DecodeScratch {
                    graph: None,
                    splits,
                    input: s.alloc::<f32>(k).expect("decode input scratch"),
                    partial: s.alloc::<f32>(n * splits).expect("decode partial scratch"),
                }
            });
        }
        let DecodeScratch {
            input,
            partial,
            graph,
            ..
        } = cache.as_mut().unwrap();
        let a = b.cuda_input(x);
        // SAFETY: the output transform initializes all n elements before return.
        let mut out = unsafe { s.alloc::<f32>(n) }.expect("decode output");
        let (ki, ni) = (k as i32, n as i32);
        // The three launches on `st`: recorded into this projection's own graph, or straight
        // into a whole step's capture (a graph launch cannot be captured).
        let launches = |st: &Arc<cudarc::driver::CudaStream>, input: &mut CudaSlice<f32>, partial: &mut CudaSlice<f32>, out: &mut CudaSlice<f32>| {
            unsafe {
                st
                    .launch_builder(b.func("exl3_had"))
                    .arg(a.as_ref())
                    .arg(&mut *input)
                    .arg(&self.suh)
                    .arg(&self.input)
                    .arg(&ki)
                    .arg(&0i32)
                    .launch(LaunchConfig {
                        grid_dim: ((k / 128) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("decode input transform");
                let mut launch = st.launch_builder(b.func(kernel));
                launch
                    .arg(&self.words)
                    .arg(&*input)
                    .arg(&mut *partial)
                    .arg(&ki)
                    .arg(&ni);
                if !tiled {
                    launch.arg(&self.tile_words);
                }
                launch
                    .launch(LaunchConfig {
                        grid_dim: (
                            (n as u32).div_ceil(if tiled { 16 } else { 8 }),
                            splits as u32,
                            1,
                        ),
                        block_dim: (32, if tiled { 4 } else { 8 }, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("decode GEMV");
                st
                    .launch_builder(b.func(if splits > 1 {
                        "exl3_had_reduce"
                    } else {
                        "exl3_had"
                    }))
                    .arg(&*partial)
                    .arg(&mut *out)
                    .arg(&self.svh)
                    .arg(&self.output)
                    .arg(&ni)
                    .arg(&(splits as i32))
                    .launch(LaunchConfig {
                        grid_dim: ((n / 128) as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("decode output transform");
            }
        };
        if s.graph_recording() {
            launches(s, &mut *input, &mut *partial, &mut out);
            let mut shape = x.shape().to_vec();
            *shape.last_mut().unwrap() = n;
            return b.make_tensor(out, shape);
        }
        if graph.is_none() {
            let recording = b.capture_stream();
            recording
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .expect("begin projection capture");
            // SAFETY: capture only records kernels; replay executes on the
            // owning compute stream, ordered after its input uploads. No buffer
            // access occurs on the private recording stream.
            launches(&recording, &mut *input, &mut *partial, &mut out);
            let recorded = recording
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .expect("end projection capture")
                .expect("projection graph");
            *graph = Some(ProjectionGraph::new(recorded, b.clone()));
        }
        let graph = graph.as_mut().unwrap();
        let (a_ptr, _a_guard) = a.as_ref().device_ptr(s);
        let (input_ptr, _input_guard) = input.device_ptr(s);
        let (partial_ptr, _partial_guard) = partial.device_ptr(s);
        let (out_ptr, _out_guard) = out.device_ptr_mut(s);
        let (suh_ptr, _suh_guard) = self.suh.device_ptr(s);
        let (svh_ptr, _svh_guard) = self.svh.device_ptr(s);
        let (imap_ptr, _imap_guard) = self.input.device_ptr(s);
        let (omap_ptr, _omap_guard) = self.output.device_ptr(s);
        let post = splits as i32;
        let zero = 0i32;
        let mut input_args = [
            arg_ptr(&a_ptr),
            arg_ptr(&input_ptr),
            arg_ptr(&suh_ptr),
            arg_ptr(&imap_ptr),
            arg_ptr(&ki),
            arg_ptr(&zero),
        ];
        let mut output_args = [
            arg_ptr(&partial_ptr),
            arg_ptr(&out_ptr),
            arg_ptr(&svh_ptr),
            arg_ptr(&omap_ptr),
            arg_ptr(&ni),
            arg_ptr(&post),
        ];
        graph.launch(&mut input_args, &mut output_args);
        drop(_out_guard);
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        b.make_tensor(out, shape)
    }
    fn linear_impl(&self, x: &Tensor, reference: bool, requested_splits: Option<usize>) -> Tensor {
        let b = &self.backend;
        let s = &b.stream;
        let (n, k) = (self.shape[0], self.shape[1]);
        assert_eq!(x.shape().last(), Some(&k));
        let m = x.numel() / k;
        if m == 1 && !reference {
            return self.decode_cached(x, requested_splits);
        }
        let a = b.cuda_input(x);
        // SAFETY: exl3_had writes all m*k elements before any read on this stream.
        let mut xh = unsafe { s.alloc::<f32>(m * k) }.expect("EXL3 input scratch");
        let (ki, ni) = (k as i32, n as i32);
        // SAFETY: validated dimensions/maps bound every access; all buffers
        // live through launches on this backend's ordered compute stream.
        unsafe {
            s.launch_builder(b.func("exl3_had"))
                .arg(a.as_ref())
                .arg(&mut xh)
                .arg(&self.suh)
                .arg(&self.input)
                .arg(&ki)
                .arg(&0i32)
                .launch(LaunchConfig {
                    grid_dim: ((k / 128) as u32, m as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .expect("EXL3 input transform");
        }
        let yh = if m == 1 {
            // SAFETY: the chosen GEMV writes every output/partial before its transform reads it.
            let mut y = unsafe { s.alloc::<f32>(n) }.expect("EXL3 output scratch");
            let kernel = match self.tile_words {
                32 => "exl3_gemv_32",
                48 => "exl3_gemv_48",
                56 => "exl3_gemv_56",
                64 => "exl3_gemv_64",
                80 => "exl3_gemv_80",
                96 => "exl3_gemv_96",
                _ => "exl3_gemv_generic",
            };
            let generic = kernel == "exl3_gemv_generic";
            // SAFETY: validated dimensions bound packed reads and each block's
            // output. The fallback takes the validated runtime tile width.
            unsafe {
                let mut launch = s.launch_builder(b.func(kernel));
                launch
                    .arg(&self.words)
                    .arg(&xh)
                    .arg(&mut y)
                    .arg(&ki)
                    .arg(&ni);
                if generic {
                    launch.arg(&self.tile_words);
                }
                launch
                    .launch(LaunchConfig {
                        grid_dim: ((n as u32).div_ceil(8), 1, 1),
                        block_dim: (32, 8, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("EXL3 GEMV");
            }
            b.make_tensor(y, vec![m, n])
        } else {
            // SAFETY: exl3_reconstruct writes every n*k element before GEMM reads it.
            let mut w = unsafe { s.alloc::<f32>(n * k) }.expect("EXL3 reconstruction scratch");
            // SAFETY: kernel bounds p against n*k, matching this allocation.
            unsafe {
                s.launch_builder(b.func("exl3_reconstruct"))
                    .arg(&self.words)
                    .arg(&mut w)
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&self.tile_words)
                    .launch(LaunchConfig::for_num_elems((n * k) as u32))
                    .expect("EXL3 reconstruction");
            }
            b.linear(
                &b.make_tensor(xh, vec![m, k]),
                &b.make_tensor(w, vec![n, k]),
            )
        };
        let input = b.cuda_input(&yh);
        // SAFETY: either output transform writes all m*n elements via a validated permutation.
        let mut y = unsafe { s.alloc::<f32>(m * n) }.expect("EXL3 result");
        // SAFETY: same validated block transform, with inverse output permutation.
        unsafe {
            s.launch_builder(b.func("exl3_had"))
                .arg(input.as_ref())
                .arg(&mut y)
                .arg(&self.svh)
                .arg(&self.output)
                .arg(&ni)
                .arg(&1i32)
                .launch(LaunchConfig {
                    grid_dim: ((n / 128) as u32, m as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .expect("EXL3 output transform");
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        b.make_tensor(y, shape)
    }
}

fn arg_ptr<T>(value: &T) -> *mut std::ffi::c_void {
    value as *const T as *mut std::ffi::c_void
}
// All graph access is serialized by Exl3Matrix::decode. Keep the backend/module
// alive until after graph destruction, including when the matrix is dropped.
struct ProjectionGraph {
    graph: CudaGraph,
    input_node: sys::CUgraphNode,
    output_node: sys::CUgraphNode,
    input_params: sys::CUDA_KERNEL_NODE_PARAMS,
    output_params: sys::CUDA_KERNEL_NODE_PARAMS,
    backend: Arc<CudaBackend>,
}
// SAFETY: moving the graph between threads is supported by CUDA when context
// binding and exclusive access are enforced. The enclosing mutex does both via
// launch/new; no raw handles or graph references escape this private type.
unsafe impl Send for ProjectionGraph {}
impl std::fmt::Debug for ProjectionGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProjectionGraph")
    }
}
impl ProjectionGraph {
    fn new(graph: CudaGraph, backend: Arc<CudaBackend>) -> Self {
        // SAFETY: this is our three-kernel chain. Validate its topology before
        // retaining nodes; CUDA fills the zero-initialized POD parameter structs.
        unsafe {
            let mut input_node = std::ptr::null_mut();
            let mut count = 1usize;
            sys::cuGraphGetRootNodes(graph.cu_graph(), &mut input_node, &mut count)
                .result()
                .expect("graph root");
            assert_eq!(count, 1);
            let mut output_node = input_node;
            for _ in 0..2 {
                let mut next = std::ptr::null_mut();
                count = 1;
                sys::cuGraphNodeGetDependentNodes(output_node, &mut next, &mut count)
                    .result()
                    .expect("graph dependent");
                assert_eq!(count, 1);
                output_node = next;
            }
            count = 0;
            sys::cuGraphNodeGetDependentNodes(output_node, std::ptr::null_mut(), &mut count)
                .result()
                .expect("graph leaf");
            assert_eq!(count, 0);
            let mut input_params = std::mem::zeroed();
            let mut output_params = std::mem::zeroed();
            sys::cuGraphKernelNodeGetParams_v2(input_node, &mut input_params)
                .result()
                .expect("input node parameters");
            sys::cuGraphKernelNodeGetParams_v2(output_node, &mut output_params)
                .result()
                .expect("output node parameters");
            input_params.kernelParams = std::ptr::null_mut();
            output_params.kernelParams = std::ptr::null_mut();
            Self {
                graph,
                input_node,
                output_node,
                input_params,
                output_params,
                backend,
            }
        }
    }
    fn launch(
        &mut self,
        input: &mut [*mut std::ffi::c_void; 6],
        output: &mut [*mut std::ffi::c_void; 6],
    ) {
        self.backend.ctx.bind_to_thread().expect("graph context");
        self.input_params.kernelParams = input.as_mut_ptr();
        self.output_params.kernelParams = output.as_mut_ptr();
        // SAFETY: CUDA copies the six arguments during each update. Both nodes
        // always receive the current input/output pointers before replay. These
        // buffers remain live until their same-stream work completes; the middle
        // node uses resident weights and scratch owned by the enclosing matrix.
        unsafe {
            sys::cuGraphExecKernelNodeSetParams_v2(
                self.graph.cu_graph_exec(),
                self.input_node,
                &self.input_params,
            )
            .result()
            .expect("update graph input");
            sys::cuGraphExecKernelNodeSetParams_v2(
                self.graph.cu_graph_exec(),
                self.output_node,
                &self.output_params,
            )
            .result()
            .expect("update graph output");
            result::graph::launch(self.graph.cu_graph_exec(), self.backend.stream.cu_stream())
                .expect("projection graph replay");
        }
        self.input_params.kernelParams = std::ptr::null_mut();
        self.output_params.kernelParams = std::ptr::null_mut();
    }
}

/// VENDORED-LOCAL (Qwen3.8-Flash-Next): a layer's routed experts plus its shared one (the last),
/// each SwiGLU (gate and up `hidden -> ff`, down `ff -> hidden`) in EXL3 without channel maps
/// (3- or 5-bit), stacked so the whole MoE runs on the GPU in a few launches: routing,
/// grouping by expert, then the experts.
#[derive(Debug)]
pub struct Exl3Experts {
    backend: Arc<CudaBackend>,
    experts: usize,
    hidden: usize,
    ff: usize,
    words: [CudaSlice<u32>; 3],
    offsets: [CudaSlice<u64>; 3],
    rates: [CudaSlice<i32>; 3],
    suh: [CudaSlice<f32>; 3],
    svh: [CudaSlice<f32>; 3],
    /// LoRA on the gate, up and down projections of some experts.
    lora: [Option<ExpertLora>; 3],
}

/// VENDORED-LOCAL: a LoRA on one projection of some experts: `slot_of[e]` their row in `a`
/// (`[slots, rank, k]`) and `b` (`[slots, n, rank]`), or `u32::MAX` for none.
#[derive(Debug)]
struct ExpertLora {
    slot_of: CudaSlice<u32>,
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    rank: usize,
}

impl ggml_rs::exl3::Experts for Exl3Experts {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        Exl3Experts::forward(self, x, logits, top_k)
    }
}

impl Exl3Experts {
    /// Add a LoRA to projection `which` (0 gate, 1 up, 2 down) of the experts `slot_of` names
    /// (`u32::MAX` for none): A `[slots, rank, k]`, B `[slots, n, rank]`, scaled already.
    pub fn set_lora(&mut self, which: usize, slot_of: &[u32], a: &[f32], b: &[f32], rank: usize) -> Result<(), String> {
        if which > 2 {
            return Err(format!("expert LoRA: no projection {which}"));
        }
        let (k, n) = if which == 2 { (self.ff, self.hidden) } else { (self.hidden, self.ff) };
        let slots = slot_of.iter().filter(|&&s| s != u32::MAX).count();
        if slot_of.len() != self.experts || rank == 0 || a.len() != slots * rank * k || b.len() != slots * n * rank {
            return Err(format!("expert LoRA: {} experts, rank {rank}: A {} and B {} values", slot_of.len(), a.len(), b.len()));
        }
        let s = &self.backend.stream;
        let up = |e: cudarc::driver::DriverError| format!("{e:?}");
        self.lora[which] = Some(ExpertLora { slot_of: s.clone_htod(slot_of).map_err(up)?, a: s.clone_htod(a).map_err(up)?, b: s.clone_htod(b).map_err(up)?, rank });
        Ok(())
    }

    /// `experts[e]` = its (gate, up, down); the last is the shared expert.
    pub fn upload(backend: Arc<CudaBackend>, experts: Vec<[Exl3Data; 3]>) -> Result<Self, String> {
        let first = experts.first().ok_or("no experts")?;
        // The grouping kernel gives each expert a thread of one block.
        if experts.len() > 1024 {
            return Err("at most 1024 experts".into());
        }
        let (hidden, ff) = (first[0].suh.len(), first[0].svh.len());
        let shapes = [(hidden, ff), (hidden, ff), (ff, hidden)];
        let mut words: [Vec<u32>; 3] = Default::default();
        let mut offsets: [Vec<u64>; 3] = Default::default();
        let mut rates: [Vec<i32>; 3] = Default::default();
        let mut suh: [Vec<f32>; 3] = Default::default();
        let mut svh: [Vec<f32>; 3] = Default::default();
        for expert in &experts {
            for (i, d) in expert.iter().enumerate() {
                d.validate()?;
                let (k, n) = shapes[i];
                let identity = |m: &[u32]| m.iter().enumerate().all(|(j, &v)| j as u32 == v);
                if d.suh.len() != k || d.svh.len() != n || !matches!(d.tile_words, 48 | 80) || !identity(&d.input_map) || !identity(&d.output_map) {
                    return Err("grouped experts need 3- or 5-bit EXL3 of one shape, without channel maps".into());
                }
                offsets[i].push(words[i].len() as u64);
                rates[i].push(d.tile_words as i32);
                words[i].extend_from_slice(&d.words);
                suh[i].extend_from_slice(&d.suh);
                svh[i].extend_from_slice(&d.svh);
            }
        }
        let s = &backend.stream;
        macro_rules! up { ($v:expr) => { s.clone_htod($v).map_err(|e| format!("{e:?}"))? } }
        Ok(Self {
            experts: experts.len(),
            hidden,
            ff,
            words: [up!(&words[0]), up!(&words[1]), up!(&words[2])],
            offsets: [up!(&offsets[0]), up!(&offsets[1]), up!(&offsets[2])],
            rates: [up!(&rates[0]), up!(&rates[1]), up!(&rates[2])],
            suh: [up!(&suh[0]), up!(&suh[1]), up!(&suh[2])],
            svh: [up!(&svh[0]), up!(&svh[1]), up!(&svh[2])],
            lora: [None, None, None],
            backend,
        })
    }

    pub fn experts(&self) -> usize { self.experts }

    /// The MoE of each row of `x` (`[rows, hidden]`): `logits` (`[rows, routed + 1]`, the
    /// router's then the shared expert's gate) pick each token's `top_k` experts; the shared
    /// one always runs.
    pub fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let b = &self.backend;
        let s = &b.stream;
        let (h, f, ex) = (self.hidden, self.ff, self.experts);
        let rows = x.dim(0);
        let per = top_k + 1;
        let total = rows * per;
        // Segments: each expert's rows in chunks of at most CHUNK (one block each).
        const CHUNK: usize = 16;
        let max_seg = total.div_ceil(CHUNK) + total.min(ex);
        let xin = b.cuda_input(x);
        let lin = b.cuda_input(logits);
        // SAFETY: each kernel writes every element of its outputs before a later launch on this
        // stream reads it; the routing and grouping tables bound every index.
        let (mut ae, mut aw, mut sr, mut se, mut sw, mut seg, mut nseg) = unsafe {(
            s.alloc::<u32>(total).expect("MoE scratch"), s.alloc::<f32>(total).expect("MoE scratch"),
            s.alloc::<u32>(total).expect("MoE scratch"), s.alloc::<u32>(total).expect("MoE scratch"),
            s.alloc::<f32>(total).expect("MoE scratch"), s.alloc::<u32>(3 * max_seg).expect("MoE scratch"),
            s.alloc::<u32>(1).expect("MoE scratch"),
        )};
        // SAFETY: the grouping writes each assignment's sorted row.
        let mut slot = unsafe { s.alloc::<u32>(total) }.expect("MoE scratch");
        // SAFETY: as above.
        let (mut xg, mut xu, mut yg, mut yu, mut xd, mut yd, mut unused) = unsafe {(
            s.alloc::<f32>(total * h).expect("MoE scratch"), s.alloc::<f32>(total * h).expect("MoE scratch"),
            s.alloc::<f32>(total * f).expect("MoE scratch"), s.alloc::<f32>(total * f).expect("MoE scratch"),
            s.alloc::<f32>(total * f).expect("MoE scratch"), s.alloc::<f32>(total * h).expect("MoE scratch"),
            s.alloc::<f32>(1).expect("MoE scratch"),
        )};
        let (hi, fi, ei, ti, pi, ki, ri) = (h as i32, f as i32, ex as i32, total as i32, per as i32, top_k as i32, (ex - 1) as i32);
        let (ci, si) = (CHUNK as i32, max_seg as i32);
        let one_row = (rows == 1) as i32;
        let row_blocks = |width: usize| LaunchConfig { grid_dim: ((width / 128) as u32, total as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: exl3_moe_out writes every row.
        let mut out_dev = unsafe { s.alloc::<f32>(rows * h) }.expect("MoE output");
        // SAFETY: as above; dimensions were validated at upload.
        unsafe {
            if rows <= 16 {
                // A few tokens: routing and grouping in one launch.
                let token_rows = rows as i32;
                s.launch_builder(b.func("exl3_moe_route_group"))
                    .arg(lin.as_ref()).arg(&mut ae).arg(&mut aw).arg(&ri).arg(&ki).arg(&token_rows)
                    .arg(&ci).arg(&si).arg(&mut sr).arg(&mut se).arg(&mut sw).arg(&mut seg).arg(&mut nseg).arg(&mut slot)
                    .launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: (2 * ex * 4) as u32 })
                    .expect("MoE routing and grouping");
            } else {
                s.launch_builder(b.func("exl3_moe_route"))
                    .arg(lin.as_ref()).arg(&mut ae).arg(&mut aw).arg(&ri).arg(&ki)
                    .launch(LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: ((ex - 1).next_multiple_of(32).min(1024) as u32, 1, 1), shared_mem_bytes: 0 })
                    .expect("MoE routing");
                s.launch_builder(b.func("exl3_moe_group"))
                    .arg(&ae).arg(&aw).arg(&ti).arg(&pi).arg(&ei).arg(&ci).arg(&si).arg(&mut sr).arg(&mut se).arg(&mut sw).arg(&mut seg).arg(&mut nseg).arg(&mut slot)
                    .launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: (2 * ex * 4) as u32 })
                    .expect("MoE grouping");
            }
            s.launch_builder(b.func("exl3_moe_in"))
                .arg(xin.as_ref()).arg(&sr).arg(&se).arg(&self.suh[0]).arg(&self.suh[1]).arg(&mut xg).arg(&mut xu).arg(&hi)
                .launch(LaunchConfig { grid_dim: ((h / 128) as u32, total as u32, 2), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
                .expect("MoE input transform");
            s.launch_builder(b.func("exl3_moe_tile"))
                .arg(&self.words[0]).arg(&self.words[1]).arg(&self.offsets[0]).arg(&self.offsets[1]).arg(&self.rates[0]).arg(&self.rates[1])
                .arg(&xg).arg(&xu).arg(&mut yg).arg(&mut yu).arg(&seg).arg(&nseg).arg(&si).arg(&hi).arg(&fi).arg(&one_row)
                .launch(LaunchConfig { grid_dim: ((f / 16) as u32, max_seg as u32, 2), block_dim: (32, 8, 1), shared_mem_bytes: 0 })
                .expect("MoE gate/up");
            // LoRA on gate and up: their deltas, from each assignment's token row.
            let null = 0u64;
            let delta = |which: usize| self.lora[which].as_ref().map(|l| {
                // SAFETY: exl3_moe_lora writes every row (zero for an expert without it).
                let mut d = s.alloc::<f32>(total * f).expect("MoE LoRA delta");
                let (ki, ni, rk, acc) = (h as i32, f as i32, l.rank as i32, 0i32);
                s.launch_builder(b.func("exl3_moe_lora"))
                    .arg(xin.as_ref()).arg(&sr).arg(&se).arg(&l.slot_of).arg(&l.a).arg(&l.b).arg(&null).arg(&mut d).arg(&ki).arg(&ni).arg(&rk).arg(&acc)
                    .launch(LaunchConfig { grid_dim: (total as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (l.rank * 4) as u32 })
                    .expect("MoE LoRA");
                d
            });
            let (dg, du) = (delta(0), delta(1));
            // SAFETY: exl3_moe_mid writes every element.
            let mut hk = self.lora[2].as_ref().map(|_| s.alloc::<f32>(total * f).expect("MoE LoRA activation"));
            let mut mid = s.launch_builder(b.func("exl3_moe_mid"));
            mid.arg(&yg).arg(&yu).arg(&self.svh[0]).arg(&self.svh[1]).arg(&self.suh[2]).arg(&se).arg(&mut xd).arg(&fi);
            match &dg { Some(d) => { mid.arg(d); } None => { mid.arg(&null); } }
            match &du { Some(d) => { mid.arg(d); } None => { mid.arg(&null); } }
            match &mut hk { Some(k) => { mid.arg(k); } None => { mid.arg(&null); } }
            mid.launch(row_blocks(f)).expect("MoE activation");
            s.launch_builder(b.func("exl3_moe_tile"))
                .arg(&self.words[2]).arg(&self.words[2]).arg(&self.offsets[2]).arg(&self.offsets[2]).arg(&self.rates[2]).arg(&self.rates[2])
                .arg(&xd).arg(&xd).arg(&mut yd).arg(&mut unused).arg(&seg).arg(&nseg).arg(&si).arg(&fi).arg(&hi).arg(&one_row)
                .launch(LaunchConfig { grid_dim: ((h / 16) as u32, max_seg as u32, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 })
                .expect("MoE down");
            s.launch_builder(b.func("exl3_moe_out"))
                .arg(&mut yd).arg(&self.svh[2]).arg(&se).arg(&sw).arg(&hi)
                .launch(row_blocks(h)).expect("MoE output");
            // LoRA on down: added to each assignment's weighted output.
            if let (Some(l), Some(k)) = (&self.lora[2], &hk) {
                let (ki, ni, rk, acc) = (f as i32, h as i32, l.rank as i32, 1i32);
                s.launch_builder(b.func("exl3_moe_lora"))
                    .arg(k).arg(&null).arg(&se).arg(&l.slot_of).arg(&l.a).arg(&l.b).arg(&sw).arg(&mut yd).arg(&ki).arg(&ni).arg(&rk).arg(&acc)
                    .launch(LaunchConfig { grid_dim: (total as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (l.rank * 4) as u32 })
                    .expect("MoE down LoRA");
            }
            let rows_i = rows as i32;
            s.launch_builder(b.func("exl3_moe_sum"))
                .arg(&yd).arg(&slot).arg(&mut out_dev).arg(&rows_i).arg(&hi).arg(&pi)
                .launch(LaunchConfig::for_num_elems((rows * h) as u32)).expect("MoE sum");
        }
        b.make_tensor(out_dev, vec![rows, h])
    }
}

/// VENDORED-LOCAL: a dense `n x k` matrix of f16 values, two to a word (`tensor::pack_f16`):
/// half the memory and reading of f32, and exact for weights stored as f16.
#[derive(Debug)]
pub struct HalfMatrix {
    backend: Arc<CudaBackend>,
    words: Tensor,
    shape: [usize; 2],
}
impl HalfMatrix {
    /// `values` (`n x k`, row-major; `k` even), each exactly an f16.
    pub fn upload(backend: Arc<CudaBackend>, values: &[f32], n: usize, k: usize) -> Self {
        assert!(k % 2 == 0 && values.len() == n * k, "HalfMatrix: {n} x {k} from {} values", values.len());
        let words = backend.to_device(Tensor::from_vec(ggml_rs::tensor::pack_f16(values), vec![n, k / 2]));
        Self { backend, words, shape: [n, k] }
    }
}
impl PackedLinear for HalfMatrix {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.words.numel() * 4
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        let b = &self.backend;
        let (n, k) = (self.shape[0], self.shape[1]);
        let m = x.numel() / k;
        // Many rows: a GEMM over the weights unpacked once.
        if m > 8 {
            return b.linear(x, &b.unpack_f16(&self.words));
        }
        let x_in = b.cuda_input(x);
        let w_in = b.cuda_input(&self.words);
        // SAFETY: one block per output element writes it.
        let mut y = unsafe { b.stream.alloc::<f32>(m * n) }.expect("f16 GEMV output");
        let (ki, ni) = (k as i32, n as i32);
        unsafe {
            b.stream.launch_builder(b.func("half_gemv_f32"))
                .arg(x_in.as_ref()).arg(w_in.as_ref()).arg(&mut y).arg(&ki).arg(&ni)
                .launch(LaunchConfig { grid_dim: (n as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
                .expect("f16 GEMV");
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        b.make_tensor(y, shape)
    }
}
