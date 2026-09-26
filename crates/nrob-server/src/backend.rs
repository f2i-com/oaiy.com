//! Which backend GGUF models run on.
//!
//! A CUDA build uses the cards `--devices` names (and GLM's expert tier across
//! them). Without CUDA -- `nrob-server-webgpu`, for machines that lack it -- the
//! quantized matmuls run on the first WebGPU adapter (D3D12, Vulkan or Metal)
//! with everything else on the CPU, and a machine with no adapter at all runs
//! on the CPU. `--backend` pins one.

use crate::Options;
use nrob::{Error, Result};
use std::sync::Arc;

pub(crate) struct Picked {
    pub backend: Arc<dyn ggml_rs::Backend>,
    /// Every CUDA card, for GLM's expert tier (CUDA builds on CUDA only).
    #[cfg(feature = "cuda")]
    pub cards: Option<Vec<Arc<ggml_rs_cuda::CudaBackend>>>,
    /// For the log: what runs the model.
    pub label: String,
}

fn cpu() -> Picked {
    Picked {
        backend: Arc::new(ggml_rs::CpuBackend::new()),
        #[cfg(feature = "cuda")]
        cards: None,
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
        #[cfg(feature = "cuda")]
        cards: None,
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
            return Err(Error::Arg("--backend webgpu needs a WebGPU build (nrob-server-webgpu)".into()));
        }
        "cuda" | "auto" if cfg!(feature = "cuda") => {
            #[cfg(feature = "cuda")]
            {
                let cards = llama_rs::glm5next::device::open_cards(devices).map_err(|e| Error::Arg(e.to_string()))?;
                let backend: Arc<dyn ggml_rs::Backend> = Arc::clone(&cards[0]) as Arc<dyn ggml_rs::Backend>;
                return Ok(Picked { backend, label: format!("CUDA ({} card(s))", cards.len()), cards: Some(cards) });
            }
            #[cfg(not(feature = "cuda"))]
            unreachable!("guarded by cfg!(feature = \"cuda\")")
        }
        "cuda" => Err(Error::Arg("--backend cuda needs the CUDA build (nrob-server)".into())),
        _ => {
            // auto without CUDA: an adapter if there is one, else the CPU.
            #[cfg(feature = "webgpu")]
            match webgpu(o) {
                Ok(p) => return Ok(p),
                Err(e) if !o.silent => eprintln!("nrob-server: WebGPU unavailable ({e}); running on the CPU"),
                Err(_) => {}
            }
            Ok(cpu())
        }
    }
}
