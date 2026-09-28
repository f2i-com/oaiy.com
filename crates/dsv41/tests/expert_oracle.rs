//! Phase A gate (docs/DEEPSEEK_V41.md): one routed expert, fetched through
//! the in-place safetensors store and run by `expert_forward`, reproduces
//! the reference model's output for the same inputs.
//!
//! Needs the real checkpoint and the oracle's golden file
//! (`tools/dsv41/oracle.py`); both are located through environment
//! variables and the test skips (passes with a note) when either is absent,
//! like the CUDA tests do without a GPU.
//!
//!   DSV41_MODEL  (default E:\deepseek\model)
//!   DSV41_GOLDEN (default E:\deepseek\golden\golden.safetensors)

use std::path::PathBuf;
use std::time::Instant;

use dsv41::expert::{expert_forward, SafetensorsExpertStore, DIM, RECORD_BYTES};
use dsv41::formats::f32_to_bf16;
use dsv41::safetensors::StIndex;
use oaiy_engine::store::WeightStore;

fn env_path(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

/// Direct reads fetch whole 4 KiB blocks, so an expert whose bytes end near
/// the end of a shard reads past EOF. For every shard, fetch the expert that
/// ends last in it through both paths and require identical bytes.
#[test]
fn experts_at_shard_ends_read_identically() {
    let model = env_path("DSV41_MODEL", r"E:\deepseek\model");
    if !model.join("config.json").exists() {
        eprintln!("skipping: checkpoint not found");
        return;
    }
    let idx = StIndex::open(&model).unwrap();
    let mut last: Vec<Option<(u64, u32, u32)>> = vec![None; idx.shard_count()];
    for name in idx.names() {
        let mut it = name.split('.');
        let (Some("layers"), Some(l), Some("ffn"), Some("experts"), Some(e)) = (it.next(), it.next(), it.next(), it.next(), it.next()) else {
            continue;
        };
        let (Ok(l), Ok(e)) = (l.parse::<u32>(), e.parse::<u32>()) else { continue };
        let t = idx.get(name).unwrap();
        let end = t.start + t.nbytes;
        if last[t.shard].is_none_or(|(best, _, _)| end > best) {
            last[t.shard] = Some((end, l, e));
        }
    }
    let buffered = SafetensorsExpertStore::open(&idx, 40, 384, false).unwrap();
    let direct = SafetensorsExpertStore::open(&idx, 40, 384, true).unwrap();
    let (mut a, mut b) = (vec![0u8; RECORD_BYTES], vec![0u8; RECORD_BYTES]);
    let mut checked = 0;
    for (end, l, e) in last.into_iter().flatten() {
        buffered.fetch(l, e, &mut a).unwrap();
        direct.fetch(l, e, &mut b).unwrap_or_else(|err| panic!("direct fetch of ({l}, {e}) ending at {end}: {err}"));
        assert!(a == b, "expert ({l}, {e}) differs between direct and buffered reads");
        checked += 1;
    }
    eprintln!("{checked} shard-final experts identical through both read paths");
    assert!(checked > 0);
}

#[test]
fn routed_expert_matches_reference() {
    let model = env_path("DSV41_MODEL", r"E:\deepseek\model");
    let golden = env_path("DSV41_GOLDEN", r"E:\deepseek\golden\golden.safetensors");
    if !model.join("config.json").exists() || !golden.exists() {
        eprintln!("skipping: checkpoint or golden file not found ({} / {})", model.display(), golden.display());
        return;
    }

    let g = StIndex::open_file(&golden).unwrap();
    let t = Instant::now();
    let idx = StIndex::open(&model).unwrap();
    eprintln!("indexed {} tensors in {:.0} ms", idx.len(), t.elapsed().as_secs_f64() * 1e3);

    for direct in [false, true] {
        let t = Instant::now();
        let store = SafetensorsExpertStore::open(&idx, 40, 384, direct).unwrap();
        eprintln!("store open (direct={direct}): {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
        let mut record = vec![0u8; RECORD_BYTES];
        let mut n = 0;
        while g.get(&format!("expert{n}.id")).is_some() {
            let id = g.read_i64(&format!("expert{n}.id")).unwrap();
            let (layer, expert) = (id[0] as u32, id[1] as u32);
            assert_eq!(store.reads_per_expert(layer, expert), 2, "weights + scales should coalesce");
            let t = Instant::now();
            store.fetch(layer, expert, &mut record).unwrap();
            let fetch_ms = t.elapsed().as_secs_f64() * 1e3;

            let x = g.read_f32(&format!("expert{n}.x")).unwrap();
            let y_ref = g.read_f32(&format!("expert{n}.y")).unwrap();
            let route = g.get(&format!("expert{n}.route_weight")).map(|_| g.read_f32(&format!("expert{n}.route_weight")).unwrap());
            let rows = x.len() / DIM;
            assert_eq!(y_ref.len(), rows * DIM);

            let (mut exact, mut max_abs, mut ref_max) = (0usize, 0.0f32, 0.0f32);
            let t = Instant::now();
            for r in 0..rows {
                let w = route.as_ref().map(|w| w[r]);
                let y = expert_forward(&record, &x[r * DIM..(r + 1) * DIM], w, 10.0);
                for (a, b) in y.iter().zip(&y_ref[r * DIM..(r + 1) * DIM]) {
                    exact += usize::from(f32_to_bf16(*a) == f32_to_bf16(*b));
                    max_abs = max_abs.max((a - b).abs());
                    ref_max = ref_max.max(b.abs());
                }
            }
            let forward_ms = t.elapsed().as_secs_f64() * 1e3 / rows as f64;
            let total = rows * DIM;
            eprintln!(
                "direct={direct} expert ({layer:2}, {expert:3}) rows={rows} fetch={fetch_ms:.1} ms forward={forward_ms:.1} ms/row  \
                 bf16-exact {exact}/{total} ({:.2}%)  max|diff| {max_abs:.3e} vs max|y| {ref_max:.3e}",
                100.0 * exact as f64 / total as f64
            );
            // Accumulation order differs (tensor-core blocks vs a scalar
            // loop), so a bf16 value can land one rounding step away; beyond
            // that, anything is a real bug.
            assert!(exact as f64 >= 0.97 * total as f64, "too few bf16-identical outputs");
            assert!(max_abs <= 2e-2 * ref_max, "max diff {max_abs} too large vs {ref_max}");
            n += 1;
        }
        assert!(n > 0, "golden file holds no expert fixtures");
    }
}
