//! How much a whole decode step's launches cost directly, and captured into a graph that is
//! updated and replayed each step (the CPU side, which is what decode waits on).
use crate::backend::CudaBackend;
use cudarc::driver::{sys, LaunchConfig, PushKernelArg};

#[test]
#[ignore = "needs a GPU"]
fn launch_versus_graph() {
    let b = CudaBackend::new(0).unwrap();
    // A stream of its own: the default one cannot be captured.
    let s = b.context().new_stream().unwrap();
    let x = s.alloc_zeros::<f32>(4096).unwrap();
    let mut l = s.alloc_zeros::<f32>(2048).unwrap();
    let mut r = s.alloc_zeros::<f32>(2048).unwrap();
    let (rows, width, a) = (1i32, 4096i32, 2048i32);
    let n = 900;
    // A small kernel, launched `n` times a step.
    let launch = |s: &std::sync::Arc<cudarc::driver::CudaStream>, x: &cudarc::driver::CudaSlice<f32>, l: &mut cudarc::driver::CudaSlice<f32>, r: &mut cudarc::driver::CudaSlice<f32>| unsafe {
        s.launch_builder(b.func("split_cols_f32")).arg(x).arg(l).arg(r).arg(&rows).arg(&width).arg(&a)
            .launch(LaunchConfig::for_num_elems(4096)).unwrap();
    };
    for round in 0..3 {
        s.synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n { launch(&s, &x, &mut l, &mut r); }
        let issue = t.elapsed();
        s.synchronize().unwrap();
        eprintln!("round {round}: direct {n} launches: issue {:.2} ms, done {:.2} ms", issue.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3);
    }
    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
    for round in 0..5 {
        s.synchronize().unwrap();
        let t = std::time::Instant::now();
        s.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED).expect("begin");
        for _ in 0..n { launch(&s, &x, &mut l, &mut r); }
        let graph = unsafe { cudarc::driver::result::stream::end_capture(s.cu_stream()) }.unwrap();
        let captured = t.elapsed();
        unsafe {
            if exec.is_null() {
                exec = cudarc::driver::result::graph::instantiate(graph, sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH).unwrap();
            } else {
                let mut info = std::mem::zeroed::<sys::CUgraphExecUpdateResultInfo>();
                sys::cuGraphExecUpdate_v2(exec, graph, &mut info).result().unwrap();
            }
        }
        let updated = t.elapsed();
        unsafe { cudarc::driver::result::graph::launch(exec, s.cu_stream()).unwrap(); }
        let issued = t.elapsed();
        s.synchronize().unwrap();
        unsafe { cudarc::driver::result::graph::destroy(graph).unwrap(); }
        eprintln!("round {round}: graph: capture {:.2} ms, update {:.2} ms, launch {:.2} ms, done {:.2} ms", captured.as_secs_f64() * 1e3,
            (updated - captured).as_secs_f64() * 1e3, (issued - updated).as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3);
    }
}
