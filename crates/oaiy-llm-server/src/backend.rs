//! Which backend GGUF models run on.
//!
//! A CUDA build uses the cards `--devices` names (and GLM's expert tier across
//! them). Without CUDA -- `oaiy-llm-server-webgpu`, for machines that lack it -- the
//! quantized matmuls run on the first WebGPU adapter (D3D12, Vulkan or Metal)
//! with everything else on the CPU, and a machine with no adapter at all runs
//! on the CPU. `--backend` pins one.

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
            return Err(Error::Arg("--backend webgpu needs a WebGPU build (oaiy-llm-server-webgpu)".into()));
        }
        "cuda" | "auto" if false => {
            unreachable!("guarded by cfg!(feature = \"cuda\")")
        }
        "cuda" => Err(Error::Arg("--backend cuda needs the CUDA build (oaiy-llm-server)".into())),
        _ => {
            // auto without CUDA: an adapter if there is one, else the CPU.
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
