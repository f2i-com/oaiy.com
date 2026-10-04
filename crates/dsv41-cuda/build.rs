//! The kernels, compiled here rather than at every start (crates/oaiy-cuda-build): the same
//! source and options NVRTC was given at run time (`--fmad=false`, so the results match the
//! dsv41 CPU reference; the sm_89+ fp8 to f16 conversion on the GPUs that have it), so the
//! engine ships without NVRTC.
fn main() {
    for f in ["src/kernels.cu", "src/ternary.cuh"] {
        println!("cargo:rerun-if-changed={f}");
    }
    let read = |f: &str| std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{f}: {e}"));
    let src = format!("{}\nextern \"C\" {{\n{}\n}}\n", read("src/kernels.cu"), read("src/ternary.cuh"));
    oaiy_cuda_build::build("dsv41", &src, &["--fmad=false"], &[(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)]);
}
