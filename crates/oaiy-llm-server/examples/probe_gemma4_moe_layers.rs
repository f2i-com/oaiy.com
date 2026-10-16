//! Probe Gemma 4 MoE per-layer KV layout. Reads `attention.head_count_kv` (i32
//! array), `attention.sliding_window_pattern` (bool array), and verifies which
//! blocks ship `attn_v.weight` vs which omit it (KV-shared via missing tensor).

use gguf::{Array, GgufFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: probe_gemma4_moe_layers <model.gguf>")?;
    let g = GgufFile::open(&path)?;

    let n_layers = g.get_u64("gemma4.block_count")? as usize;

    let kv_arr = g.metadata().get("gemma4.attention.head_count_kv")
        .and_then(|v| v.as_array());
    let kv_per_layer: Vec<i32> = match kv_arr {
        Some(Array::I32(v)) => v.clone(),
        Some(Array::U32(v)) => v.iter().map(|&x| x as i32).collect(),
        _ => vec![],
    };

    let swa_arr = g.metadata().get("gemma4.attention.sliding_window_pattern")
        .and_then(|v| v.as_array());
    let swa_per_layer: Vec<bool> = match swa_arr {
        Some(Array::Bool(v)) => v.clone(),
        _ => vec![],
    };

    println!("layers: {n_layers}");
    println!("{:>5}  {:>14}  {:>10}  {:>10}", "layer", "head_count_kv", "is_swa", "has_attn_v");
    for i in 0..n_layers {
        let kv = kv_per_layer.get(i).copied().unwrap_or(-1);
        let sw = swa_per_layer.get(i).copied();
        let sw_str = match sw { Some(true) => "true", Some(false) => "false", None => "?" };
        let has_v = g.tensor_by_name(&format!("blk.{i}.attn_v.weight")).is_some();
        println!("{i:>5}  {kv:>14}  {sw_str:>10}  {has_v:>10}");
    }
    Ok(())
}
