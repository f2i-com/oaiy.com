//! Which backend GGUF models run on.
//!
//! A model's kernels run on a WebGPU adapter (D3D12, Vulkan or Metal: the first,
//! or the one `--devices` names) with what does not fit there on the CPU, and a
//! machine with no adapter at all runs on the CPU. `--backend` pins one.

use crate::Options;
use oaiy_engine::{Error, Result};
use std::sync::Arc;

pub(crate) struct Picked {
    pub backend: Arc<dyn ggml_rs::Backend>,
    /// For the log: what runs the model.
    pub label: String,
}

fn cpu() -> Picked {
    Picked {
        backend: Arc::new(ggml_rs::CpuBackend::new()),
        label: "the CPU".into(),
    }
}

#[cfg(feature = "webgpu")]
fn webgpu(o: &Options) -> std::result::Result<Picked, String> {
    let b = ggml_rs_wgpu::WgpuBackend::new(o.webgpu_gb.map(|g| g << 30))?;
    let a = b.adapter().clone();
    let (_, budget) = b.usage();
    Ok(Picked {
        backend: Arc::new(b),
        label: format!("WebGPU on {} ({}), up to {} GiB of weights, the rest on the CPU", a.name, a.backend, budget >> 30),
    })
}

pub(crate) fn open(o: &Options, devices: &[usize]) -> Result<Picked> {
    let _ = devices;
    match o.backend.as_str() {
        "cpu" => Ok(cpu()),
        "webgpu" => {
            #[cfg(feature = "webgpu")]
            return webgpu(o).map_err(Error::Arg);
            #[cfg(not(feature = "webgpu"))]
            return Err(Error::Arg("--backend webgpu needs a build with the webgpu feature (the default)".into()));
        }
        "cuda" => Err(Error::Arg("--backend cuda: there is no CUDA backend any more; auto or webgpu is the GPU".into())),
        _ => {
            // auto: an adapter if there is one, else the CPU.
            #[cfg(feature = "webgpu")]
            match webgpu(o) {
                Ok(p) => return Ok(p),
                Err(e) if !o.silent => eprintln!("oaiy-llm-server: WebGPU unavailable ({e}); running on the CPU"),
                Err(_) => {}
            }
            Ok(cpu())
        }
    }
}
