//! Which backend GGUF models run on.
//!
//! A model's kernels run on a WebGPU adapter (D3D12, Vulkan or Metal) with what
//! does not fit there on the CPU, and a machine with no adapter at all runs on
//! the CPU. `--backend` pins one.
//!
//! Which GPUs: the ones `--devices` (or a model's `--devices-for`) names, by
//! their number as nvidia-smi counts them, the first carrying the model and the
//! rest what a model spreads over more than one ([`others`]); with none named,
//! the first adapter and every other discrete GPU. A host that keeps a card
//! for something else (the studio's media jobs, the voice service) says so by
//! naming the ones this server may use.

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

/// What a list of devices asks for: the first GPU's number (None: the adapter WebGPU prefers, or the one
/// `OAIY_WEBGPU_ADAPTER` names), and the others a model may spread over (None: every other discrete GPU).
pub(crate) fn plan(devices: &[usize]) -> (Option<usize>, Option<Vec<usize>>) {
    match devices.split_first() {
        None => (None, None),
        Some((&first, rest)) => {
            // (a GPU named twice is one GPU)
            let mut seen = vec![first];
            let rest: Vec<usize> = rest.iter().copied().filter(|d| !seen.contains(d) && { seen.push(*d); true }).collect();
            (Some(first), Some(rest))
        }
    }
}

#[cfg(feature = "webgpu")]
fn webgpu(o: &Options, devices: &[usize]) -> std::result::Result<Picked, String> {
    let budget = o.webgpu_gb.map(|g| g << 30);
    let b = match plan(devices).0 {
        Some(first) => ggml_rs_wgpu::WgpuBackend::nth(first, budget).map_err(|e| format!("--devices: {e}"))?,
        None => ggml_rs_wgpu::WgpuBackend::new(budget)?,
    };
    let a = b.adapter().clone();
    let (_, budget) = b.usage();
    Ok(Picked {
        backend: Arc::new(b),
        // (a GPU whose memory is the computer's says so: what it holds comes out of the RAM the rest would use)
        label: format!("WebGPU on {} ({}{}), up to {} GiB of weights, the rest on the CPU", a.name, a.backend, if a.unified { ", the computer's own memory" } else { "" }, budget >> 30),
    })
}

/// The other GPUs a model that `first` carries may spread over: the rest of `devices` where it names any (each
/// opened by its number; one that is `first`'s own card, or cannot be opened, left out), else every other discrete
/// GPU. `OAIY_NO_SPLIT`: none.
#[cfg(feature = "webgpu")]
pub(crate) fn others(o: &Options, devices: &[usize], first: &ggml_rs_wgpu::WgpuBackend) -> Vec<ggml_rs_wgpu::WgpuBackend> {
    if std::env::var_os("OAIY_NO_SPLIT").is_some() {
        return Vec::new();
    }
    let budget = o.webgpu_gb.map(|g| g << 30);
    match plan(devices).1 {
        None => first.others(budget),
        Some(rest) => {
            let mut taken = vec![first.adapter().pci_bus_id.clone()];
            rest.into_iter()
                .filter_map(|d| ggml_rs_wgpu::WgpuBackend::nth(d, budget).ok())
                .filter(|b| {
                    let at = b.adapter().pci_bus_id.clone();
                    !taken.contains(&at) && {
                        taken.push(at);
                        true
                    }
                })
                .collect()
        }
    }
}

pub(crate) fn open(o: &Options, devices: &[usize]) -> Result<Picked> {
    // (the devices name GPUs: a build without WebGPU has none to name)
    #[cfg(not(feature = "webgpu"))]
    let _ = devices;
    match o.backend.as_str() {
        "cpu" => Ok(cpu()),
        "webgpu" => {
            #[cfg(feature = "webgpu")]
            return webgpu(o, devices).map_err(Error::Arg);
            #[cfg(not(feature = "webgpu"))]
            return Err(Error::Arg("--backend webgpu needs a build with the webgpu feature (the default)".into()));
        }
        "cuda" => Err(Error::Arg("--backend cuda: there is no CUDA backend any more; auto or webgpu is the GPU".into())),
        _ => {
            // auto: an adapter if there is one, else the CPU.
            #[cfg(feature = "webgpu")]
            match webgpu(o, devices) {
                Ok(p) => return Ok(p),
                Err(e) if !o.silent => eprintln!("oaiy-llm-server: WebGPU unavailable ({e}); running on the CPU"),
                Err(_) => {}
            }
            Ok(cpu())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::plan;

    #[test]
    fn a_list_of_devices_names_the_first_gpu_and_the_others() {
        // none named: the adapter WebGPU prefers, and every other discrete GPU
        assert_eq!(plan(&[]), (None, None));
        // one named: that GPU and no other (a host kept the rest for something else)
        assert_eq!(plan(&[0]), (Some(0), Some(vec![])));
        assert_eq!(plan(&[1]), (Some(1), Some(vec![])));
        // several: the first carries the model, the rest in their order, each once
        assert_eq!(plan(&[1, 0]), (Some(1), Some(vec![0])));
        assert_eq!(plan(&[0, 1, 1, 0, 2]), (Some(0), Some(vec![1, 2])));
    }
}
