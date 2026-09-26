//! `nrob-server-webgpu`: the same program, built without CUDA so it starts on
//! machines that lack it (see Cargo.toml). GGUF models run on WebGPU, else the CPU.

#![forbid(unsafe_code)]

#[path = "main.rs"]
mod server;

fn main() {
    server::main()
}
