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
        if graph.is_none() {
            let recording = b.capture_stream();
            recording
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .expect("begin projection capture");
            // SAFETY: capture only records kernels; replay executes on the
            // owning compute stream, ordered after its input uploads. No buffer
            // access occurs on the private recording stream.
            unsafe {
                recording
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
                let mut launch = recording.launch_builder(b.func(kernel));
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
                recording
                    .launch_builder(b.func(if splits > 1 {
                        "exl3_had_reduce"
                    } else {
                        "exl3_had"
                    }))
                    .arg(&*partial)
                    .arg(&mut out)
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
