//! The kernels, compiled here rather than at every start (crates/oaiy-cuda-build): the same
//! source and options NVRTC was given at run time, for each GPU architecture the backend
//! targeted there (DP4A for EXL3 from sm_61), so the engine ships without NVRTC.
fn main() {
    let files = ["src/kernels.cu", "src/exl3.cu", "src/long_attention.cu"];
    for f in files {
        println!("cargo:rerun-if-changed={f}");
    }
    let read = |f: &str| std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
    let src = format!("{}\n{}\n{}", read(files[0]), read(files[1]), read(files[2]));
    oaiy_cuda_build::build("ggml", &src, &[], &[(6, 1), (7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]);
}
