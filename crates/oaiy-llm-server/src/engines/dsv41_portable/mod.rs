//! DeepSeek-V4.1 without CUDA: the CPU model (`crate::dsv41::model`, the reference the CUDA path is tested against) with its
//! dense trunk on the WebGPU adapter. Measured on the CPU alone, a prompt is compute-bound and its attention (the fp8
//! trunk's projections among it) costs more than its experts, and a warm decode step spends a flat second there
//! (docs/DEEPSEEK_V41.md, "The CPU model, measured"): so the trunk goes to the GPU, every fp8 and bf16 matrix the
//! budget holds (`ggml_rs_wgpu::dense`), the activation still quantized and the result still rounded by the CPU model
//! as the reference does. The routed experts go there too ([`WgpuExperts`]): a prompt's busy ones through slots its
//! records are uploaded into, and the ones used most kept there between passes, a decode step's computed there while
//! the CPU reads and computes the rest.

use std::sync::{Arc, Mutex};

use crate::dsv41::linear::{DenseKernel, Weight};
use ggml_rs_wgpu::dense::{DenseData, DenseGpu, RecordSlots};
use ggml_rs_wgpu::WgpuBackend;

#[cfg(test)]
mod attend_tests;
mod engine;
mod exl3;
mod experts;
mod resident;
mod saved;
#[cfg(test)]
mod tests;
mod trunk;
mod usage;

pub(crate) use engine::Engine;
pub(crate) use exl3::WgpuExl3Experts;
pub(crate) use experts::{WgpuExperts, ALONE};
pub(crate) use trunk::{layer_homes, offload, offload_split};
pub(crate) use usage::{read_usage, write_usage};

/// DSpark's drafter of the checkpoint at `dir` (whose trunk `model` is), if it has one, ready to draft: its dense
/// weights on `gpu` beside the trunk's, and an EXL3 checkpoint's 384 experts held on `cards` in slots of their own
/// (5 GB), read in now; an fp4 checkpoint's run on the CPU from a RAM cache that may hold them all (7 GB, taken as
/// they are first read, as the trunk's RAM tier takes its records: what cannot be had is read around). To be made
/// before the trunk's experts take what the cards have left. With what it took, said.
pub(crate) fn load_drafter(dir: &std::path::Path, model: &crate::dsv41::model::Model, gpu: &WgpuBackend, cards: &[Arc<WgpuBackend>]) -> oaiy_engine::Result<Option<(crate::dsv41::dspark::Drafter, String)>> {
    let Some(spec) = model.cfg.dspark.clone() else { return Ok(None) };
    let idx = crate::dsv41::safetensors::StIndex::open(dir)?;
    let exl3 = matches!(model.expert_format(), crate::dsv41::expert::ExpertFormat::Exl3 { .. });
    let records = if exl3 { 16 } else { spec.layers * spec.n_routed_experts };
    let Some(mut d) = crate::dsv41::dspark::Drafter::load(&idx, &model.cfg, model.max_seq(), records, true)? else { return Ok(None) };
    let (count, bytes) = trunk::offload_drafter(&mut d, gpu, exl3);
    let mut said = format!("DSpark's drafter: {count} dense matrices ({:.1} GB) on the GPU", bytes as f64 / 1e9);
    if exl3 {
        match WgpuExl3Experts::new_within(&d.experts.format, cards, spec.n_routed_experts, Some(spec.layers * spec.n_routed_experts), None) {
            Some(k) => {
                let started = std::time::Instant::now();
                while k.fill_some(d.experts.store.as_ref(), &d.experts.cache, None, 64) {}
                said += &format!(", its {} experts held there (read in {:.1} s)", k.tier().1, started.elapsed().as_secs_f64());
                d.set_experts_kernel(Some(Arc::new(k) as Arc<dyn crate::dsv41::expert::ExpertsKernel>));
            }
            None => said += ", its experts on the CPU (no room for them on the GPUs)",
        }
    }
    Ok(Some((d, said)))
}
