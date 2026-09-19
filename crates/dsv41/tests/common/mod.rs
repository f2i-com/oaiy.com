//! Shared setup for the golden-file tests.
//!
//!   DSV41_MODEL        checkpoint directory (default E:\deepseek\model)
//!   DSV41_GOLDEN_DIR   oracle output directory (default E:\deepseek\golden)
//!   DSV41_GOLDEN_NAME  golden file in it (default golden.safetensors)

use std::path::PathBuf;

use dsv41::model::{Model, ModelOptions};
use dsv41::safetensors::StIndex;

pub fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

/// The golden file and a model configured like the oracle run that wrote it,
/// or `None` (test skips) when the checkpoint or golden files are absent.
pub fn setup(opts: ModelOptions) -> Option<(StIndex, Model)> {
    let model_dir = env_path("DSV41_MODEL", r"E:\deepseek\model");
    let golden_dir = env_path("DSV41_GOLDEN_DIR", r"E:\deepseek\golden");
    let name = std::env::var("DSV41_GOLDEN_NAME").unwrap_or_else(|_| "golden.safetensors".into());
    let (golden, meta) = (golden_dir.join(&name), golden_dir.join("engram_meta.safetensors"));
    if !model_dir.join("config.json").exists() || !golden.exists() || !meta.exists() {
        eprintln!("skipping: checkpoint or golden files not found");
        return None;
    }
    let g = StIndex::open_file(&golden).unwrap();
    let t = std::time::Instant::now();
    let mut model = Model::load(&model_dir, &meta, &opts).unwrap();
    // test-only overrides the oracle recorded in the golden file's metadata
    dsv41::golden::apply_overrides(&mut model, &g).unwrap();
    eprintln!(
        "{name}: model loaded in {:.1}s (index_topk {}, candidate_topk_blocks {})",
        t.elapsed().as_secs_f64(),
        model.cfg.index_topk,
        model.cfg.candidate_topk_blocks
    );
    Some((g, model))
}
