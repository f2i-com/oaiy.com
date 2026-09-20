//! Tests for the streaming expert backend.
//!
//! The synthetic fixtures build a tiny stacked-experts GGUF in memory with
//! `gguf::reader::write_to_vec`, attach a Vec-backed `TensorBytes` source, and check
//! the store's records against the raw tensor slices. The equivalence test
//! runs one expert's FFN through the resident path (`split_stacked_experts`
//! + `FfnPair::from_halves`) and through `LayerStream::expert_weights` and
//! demands bitwise-identical outputs — same bytes, same kernels.
//!
//! The 30B end-to-end equivalence test is gated on the real model files
//! existing under E:/models (see `money_test_equivalence_30b`).

use std::collections::BTreeMap;
use std::sync::Arc;

use ggml_quants::GgmlType;
use gguf::{GgufFile, TensorBytes, TensorInfo, Value};

use crate::Model;

use nrob::ecache::Ecache;
use nrob::store::WeightStore;
use nrob::types::CachePolicy;

use super::*;

/// 2 layers, 4 experts, hidden 32, ff 64, Q8_0 everywhere.
pub(crate) const N_LAYERS: usize = 2;
pub(crate) const N_EXPERTS: usize = 4;
pub(crate) const HIDDEN: usize = 32;
pub(crate) const FF: usize = 64;

/// Q8_0: 32 elements per block, 34 bytes per block (f16 delta + 32 i8).
const Q8_0_BLOCK: usize = 32;
const Q8_0_TYPE: usize = 34;

fn q8_0_bytes(numel: usize) -> usize {
    (numel / Q8_0_BLOCK) * Q8_0_TYPE
}

/// Deterministic pseudo-random bytes (no external crates in tests either).
pub(crate) fn fill_bytes(n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 24) as u8
        })
        .collect()
}

/// Vec-backed `TensorBytes` over the synthetic data section — the in-memory
/// analogue of `gguf::FileSource`.
#[derive(Debug)]
struct VecSource {
    data: Vec<u8>,
}

impl TensorBytes for VecSource {
    fn read_tensor(&self, info: &TensorInfo) -> gguf::Result<Vec<u8>> {
        let s = info.offset as usize;
        Ok(self.data[s..s + info.nbytes() as usize].to_vec())
    }

    fn read_range(&self, data_offset: u64, dst: &mut [u8]) -> gguf::Result<()> {
        let s = data_offset as usize;
        if s + dst.len() > self.data.len() {
            return Err(gguf::GgufError::Truncated {
                offset: self.data.len() as u64,
                needed: (s + dst.len() - self.data.len()) as u64,
            });
        }
        dst.copy_from_slice(&self.data[s..s + dst.len()]);
        Ok(())
    }
}

pub(crate) struct Fixture {
    pub(crate) file: GgufFile,
    /// Expected record bytes per (layer, expert): gate || up || down.
    pub(crate) records: Vec<Vec<Vec<u8>>>,
    pub(crate) rec_bytes: usize,
}

/// Build the shared synthetic model pieces: metadata, per-tensor bytes, and
/// the per-(layer, expert) records (gate || up || down) the store must
/// reproduce. Used by both the in-memory (`VecSource`) and the on-disk
/// `.gguf` fixtures.
pub(crate) fn fixture_tensors() -> (
    BTreeMap<String, Value>,
    Vec<(TensorInfo, Vec<u8>)>,
    Vec<Vec<Vec<u8>>>,
    usize,
) {
    let gate_per = q8_0_bytes(FF * HIDDEN);
    let up_per = q8_0_bytes(FF * HIDDEN);
    let down_per = q8_0_bytes(HIDDEN * FF);
    let rec_bytes = gate_per + up_per + down_per;

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        Value::String("qwen3moe".to_string()),
    );
    metadata.insert("qwen3moe.expert_count".to_string(), Value::U32(N_EXPERTS as u32));

    let mut tensors: Vec<(TensorInfo, Vec<u8>)> = Vec::new();
    let mut records = vec![vec![vec![0u8; rec_bytes]; N_EXPERTS]; N_LAYERS];
    for l in 0..N_LAYERS {
        // (part, per-expert bytes, gguf fastest dim, gguf second dim)
        for (part, per, d0, d1) in [
            ("gate", gate_per, HIDDEN, FF),
            ("up", up_per, HIDDEN, FF),
            ("down", down_per, FF, HIDDEN),
        ] {
            let name = format!("blk.{l}.ffn_{part}_exps.weight");
            let bytes = fill_bytes(per * N_EXPERTS, (l * 100 + d0) as u32 + part.len() as u32);
            for e in 0..N_EXPERTS {
                let dst_off = match part {
                    "gate" => 0,
                    "up" => gate_per,
                    _ => gate_per + up_per,
                };
                records[l][e][dst_off..dst_off + per]
                    .copy_from_slice(&bytes[e * per..(e + 1) * per]);
            }
            tensors.push((
                TensorInfo {
                    name,
                    // GGUF dims are fastest-varying-first: [cols, rows, n_experts].
                    shape: vec![d0 as u64, d1 as u64, N_EXPERTS as u64],
                    dtype: GgmlType::Q8_0,
                    offset: 0, // write_to_vec re-lays-out offsets below
                },
                bytes,
            ));
        }
    }
    (metadata, tensors, records, rec_bytes)
}

/// Build a tiny qwen3moe GGUF with stacked Q8_0 expert tensors and return it
/// with the per-expert records the store must reproduce.
pub(crate) fn stacked_fixture() -> Fixture {
    let (metadata, tensors, records, rec_bytes) = fixture_tensors();

    let raw = gguf::reader::write_to_vec(&metadata, &tensors, 32).expect("write gguf");
    let probe = GgufFile::from_bytes(raw.clone()).expect("parse gguf");
    let data_start = probe.tensor_data_start() as usize;
    let source = Arc::new(VecSource { data: raw[data_start..].to_vec() });
    let file = GgufFile::from_bytes_with_source(raw, source).expect("parse gguf with source");

    // Sanity: write_to_vec packed the tensors contiguously in order, so the
    // source's data section lines up with `records`.
    Fixture { file, records, rec_bytes }
}

pub(crate) fn make_store(fx: &Fixture) -> GgufExpertStore {
    let layout = ExpertLayout::resolve(&fx.file, N_LAYERS, N_EXPERTS).expect("resolve layout");
    assert_eq!(layout.record_bytes(), fx.rec_bytes);
    GgufExpertStore::new(fx.file.clone(), layout).expect("store")
}

#[test]
fn fetch_returns_gate_up_down_record() {
    let fx = stacked_fixture();
    let store = make_store(&fx);
    assert_eq!(store.record_bytes(), fx.rec_bytes);
    assert_eq!(store.shape(), (N_LAYERS as u32, N_EXPERTS as u32));

    for l in 0..N_LAYERS as u32 {
        for e in 0..N_EXPERTS as u32 {
            let mut dst = vec![0u8; fx.rec_bytes];
            store.fetch(l, e, &mut dst).expect("fetch");
            assert_eq!(
                dst,
                fx.records[l as usize][e as usize],
                "record ({l}, {e}) mismatch"
            );
        }
    }
}

#[test]
fn fetch_rejects_out_of_range_and_bad_dst() {
    let fx = stacked_fixture();
    let store = make_store(&fx);

    let mut dst = vec![0u8; fx.rec_bytes];
    assert!(store.fetch(N_LAYERS as u32, 0, &mut dst).is_err());
    assert!(store.fetch(0, N_EXPERTS as u32, &mut dst).is_err());
    let mut short = vec![0u8; fx.rec_bytes - 1];
    assert!(store.fetch(0, 0, &mut short).is_err());
}

#[test]
fn disabled_cache_passes_through() {
    let fx = stacked_fixture();
    let store = make_store(&fx);
    let cache = Ecache::new(0, fx.rec_bytes, CachePolicy::Lfru);
    assert!(!cache.is_enabled());

    let mut dst = vec![0u8; fx.rec_bytes];
    cache.get(1, 2, &store, &mut dst).expect("get");
    assert_eq!(dst, fx.records[1][2]);
    cache.get(1, 2, &store, &mut dst).expect("get again");
    let st = cache.stats();
    assert_eq!(st.misses, 2);
    assert_eq!(st.hits, 0);
}

#[test]
fn cached_second_get_hits_and_matches() {
    let fx = stacked_fixture();
    let store = make_store(&fx);
    let cache = Ecache::new(4 * fx.rec_bytes, fx.rec_bytes, CachePolicy::Lfru);

    let mut dst = vec![0u8; fx.rec_bytes];
    cache.get(0, 3, &store, &mut dst).expect("get 1");
    cache.get(0, 3, &store, &mut dst).expect("get 2");
    assert_eq!(dst, fx.records[0][3]);
    let st = cache.stats();
    assert_eq!(st.misses, 1);
    assert_eq!(st.hits, 1);
}

/// Bitwise-equal comparison that treats NaN == NaN: random fixture bytes
/// dequantize to garbage (including NaN deltas), and NaN != NaN under `==`.
pub(crate) fn assert_f32_bitwise(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length differs");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
            "{what}: element {i} differs: {x} vs {y}"
        );
    }
}

/// The core equivalence claim: an expert rebuilt from a streamed record is
/// the same `FfnPair` + down `Weight` the resident loader builds by slicing
/// the stacked tensors, and produces bitwise-identical outputs.
#[test]
fn streamed_expert_matches_resident_expert() {
    let fx = stacked_fixture();
    let store = make_store(&fx);
    let shared = StreamShared::new(store, 8 * fx.rec_bytes, 0).expect("shared");
    let backend = ggml_rs::default_backend();

    let idx = crate::loader::TensorIndex::new(&fx.file);
    let x = Tensor::from_vec(
        (0..HIDDEN).map(|i| (i as f32 - 16.0) / 16.0).collect(),
        vec![1, HIDDEN],
    );

    for l in 0..N_LAYERS {
        // Resident path: slice the stacked tensors, fuse gate+up.
        let gate_stacked = idx
            .take_weight(&format!("blk.{l}.ffn_gate_exps.weight"), &[])
            .expect("gate");
        let up_stacked = idx
            .take_weight(&format!("blk.{l}.ffn_up_exps.weight"), &[])
            .expect("up");
        let down_stacked = idx
            .take_weight(&format!("blk.{l}.ffn_down_exps.weight"), &[])
            .expect("down");
        let gates = crate::qwen3moe::split_stacked_experts(gate_stacked, N_EXPERTS).unwrap();
        let ups = crate::qwen3moe::split_stacked_experts(up_stacked, N_EXPERTS).unwrap();
        let downs = crate::qwen3moe::split_stacked_experts(down_stacked, N_EXPERTS).unwrap();

        let stream = shared.layer(l as u32);
        let iter = gates.into_iter().zip(ups.into_iter()).zip(downs.into_iter());
        for (e, ((g, u), resident_down)) in iter.enumerate() {
            let resident_pair = FfnPair::from_halves(g, u);
            let (stream_pair, stream_down) =
                stream.expert_weights(e as u32).expect("stream expert");

            let r_act = resident_pair.swiglu(&*backend, &x);
            let s_act = stream_pair.swiglu(&*backend, &x);
            assert_f32_bitwise(
                r_act.to_host().data(),
                s_act.to_host().data(),
                &format!("layer {l} expert {e}: swiglu"),
            );
            let r_out = resident_down.linear(&*backend, &r_act);
            let s_out = stream_down.linear(&*backend, &s_act);
            assert_f32_bitwise(
                r_out.to_host().data(),
                s_out.to_host().data(),
                &format!("layer {l} expert {e}: down projection"),
            );
        }
    }
    assert!(shared.error().is_none());
    // 2 layers x 4 experts, all distinct: all misses, no evictions (8 slots).
    let st = shared.cache_stats();
    assert_eq!(st.misses, (N_LAYERS * N_EXPERTS) as u64);
}

#[test]
fn stream_shared_refuses_tiny_budget() {
    let fx = stacked_fixture();
    let store = make_store(&fx);
    assert!(StreamShared::new(store, fx.rec_bytes * (MIN_CACHE_RECORDS - 1), 0).is_err());
}

/// A non-uniform model (layer 1's experts differ in size from layer 0's)
/// must be rejected at resolve time, not mis-strided at fetch time.
#[test]
fn resolve_rejects_nonuniform_layers() {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "general.architecture".to_string(),
        Value::String("qwen3moe".to_string()),
    );
    let per = q8_0_bytes(FF * HIDDEN);
    let mut tensors: Vec<(TensorInfo, Vec<u8>)> = Vec::new();
    for l in 0..2usize {
        // Layer 1 gets 2x the ff dim on its gate tensor.
        let rows = if l == 1 { 2 * FF } else { FF };
        let bytes = fill_bytes(q8_0_bytes(rows * HIDDEN) * N_EXPERTS, l as u32 + 1);
        tensors.push((
            TensorInfo {
                name: format!("blk.{l}.ffn_gate_exps.weight"),
                shape: vec![HIDDEN as u64, rows as u64, N_EXPERTS as u64],
                dtype: GgmlType::Q8_0,
                offset: 0,
            },
            bytes,
        ));
        for part in ["up", "down"] {
            tensors.push((
                TensorInfo {
                    name: format!("blk.{l}.ffn_{part}_exps.weight"),
                    shape: vec![HIDDEN as u64, FF as u64, N_EXPERTS as u64],
                    dtype: GgmlType::Q8_0,
                    offset: 0,
                },
                fill_bytes(per * N_EXPERTS, l as u32 + 7),
            ));
        }
    }
    let raw = gguf::reader::write_to_vec(&metadata, &tensors, 32).expect("write gguf");
    let file = GgufFile::from_bytes(raw).expect("parse");
    assert!(ExpertLayout::resolve(&file, 2, N_EXPERTS).is_err());
}

/// Money test, gated on the real artifacts: greedy-generate a fixed prompt
/// through (b) the fully-resident path and (c) the streaming path with a
/// small cache; token ids must be identical. Run explicitly with:
///   cargo test -p llama-rs --release money_test -- --ignored --nocapture
#[test]
#[ignore]
fn money_test_equivalence_30b() {
    let path = "E:/models/Qwen3-30B-A3B-Q4_K_M.gguf";
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: {path} not present");
        return;
    }
    let backend = ggml_rs::default_backend();
    let prompt = "The capital of France is";
    let n = 16usize;

    let g = GgufFile::open(path).expect("open gguf");
    let resident = Model::load(&g, backend.clone()).expect("resident open");
    let ids = resident.tokenizer().encode(prompt, true).expect("encode");
    let toks_resident: Vec<u32> = resident
        .generate(&ids, crate::SampleParams::greedy(), n)
        .collect();
    drop(resident);
    drop(g);

    let budget = 4u64 << 30; // 4 GiB total: ~resident trunk + ~2.5 GiB cache
    let streaming = Model::open_streaming(path, backend, budget).expect("streaming open");
    let toks_stream: Vec<u32> = streaming
        .generate(&ids, crate::SampleParams::greedy(), n)
        .collect();

    assert!(
        streaming.expert_stream_error().is_none(),
        "streaming error: {:?}",
        streaming.expert_stream_error()
    );
    assert_eq!(toks_resident, toks_stream, "token ids diverge");
    let st = streaming.expert_cache_stats().expect("cache stats");
    eprintln!(
        "tokens: {toks_stream:?}\ncache: {} hits / {} misses ({:.1}% hit), {:.2} GiB read",
        st.hits,
        st.misses,
        100.0 * st.hits as f64 / (st.hits + st.misses).max(1) as f64,
        st.bytes_read as f64 / (1u64 << 30) as f64,
    );
}

/* ---- on-disk fixtures: experts streamed from a real .gguf file --------- */

/// On-disk fixture: the same synthetic model as `stacked_fixture`, written
/// to a temporary `.gguf` and opened with [`GgufFile::open_streaming`], so
/// expert bytes come from positioned reads of the file ([`gguf::FileSource`]).
struct FileFixture {
    pub(crate) file: GgufFile,
    pub(crate) records: Vec<Vec<Vec<u8>>>,
    pub(crate) rec_bytes: usize,
    /// The whole GGUF data section, for byte-level read_range checks.
    data: Vec<u8>,
    path: std::path::PathBuf,
}

impl Drop for FileFixture {
    fn drop(&mut self) {
        // The source may still hold the file open (another clone of the
        // GgufFile); the delete is best-effort.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Temp-file uniquifier: tests in this binary run concurrently in one
/// process, so a pid alone is not enough.
static FILE_FIXTURE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn file_fixture() -> FileFixture {
    let (metadata, tensors, records, rec_bytes) = fixture_tensors();
    let raw = gguf::reader::write_to_vec(&metadata, &tensors, 32).expect("write gguf");
    let probe = GgufFile::from_bytes(raw.clone()).expect("parse gguf");
    let data_start = probe.tensor_data_start() as usize;

    let path = std::env::temp_dir().join(format!(
        "nrob_llama_stream_{}_{}.gguf",
        std::process::id(),
        FILE_FIXTURE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(&path, &raw).expect("write fixture");
    let file = GgufFile::open_streaming(&path).expect("open streaming");
    FileFixture {
        file,
        records,
        rec_bytes,
        data: raw[data_start..].to_vec(),
        path,
    }
}

fn make_file_store(fx: &FileFixture) -> GgufExpertStore {
    let layout = ExpertLayout::resolve(&fx.file, N_LAYERS, N_EXPERTS).expect("resolve layout");
    GgufExpertStore::new(fx.file.clone(), layout).expect("store")
}

/// Every record read back from the file is byte-identical to the one the
/// fixture was built from.
#[test]
fn file_fetch_matches_records() {
    let fx = file_fixture();
    let store = make_file_store(&fx);
    assert_eq!(store.record_bytes(), fx.rec_bytes);
    for l in 0..N_LAYERS as u32 {
        for e in 0..N_EXPERTS as u32 {
            let mut dst = vec![0u8; fx.rec_bytes];
            store.fetch(l, e, &mut dst).expect("file fetch");
            assert_eq!(dst, fx.records[l as usize][e as usize], "record ({l}, {e}) mismatch");
        }
    }
}

/// The file source serves byte-identical ranges anywhere in the data
/// section (tensor starts, middles, ranges crossing tensor boundaries) and
/// refuses reads past the end of the file.
#[test]
fn file_source_read_range_byte_exact() {
    let fx = file_fixture();
    let source = fx.file.tensor_source().expect("streaming file has a source");
    let n = fx.data.len();
    let per = q8_0_bytes(FF * HIDDEN);

    let check = |off: usize, len: usize| {
        let mut dst = vec![0u8; len];
        source
            .read_range(off as u64, &mut dst)
            .unwrap_or_else(|e| panic!("read_range [{off}, +{len}): {e}"));
        assert_eq!(dst, &fx.data[off..off + len], "range [{off}, +{len})");
    };
    check(0, n); // whole data section
    check(0, 1); // first byte
    check(n - 1, 1); // last byte
    check(1, n - 2); // unaligned both ends
    check(per - 5, 2 * per + 10); // crosses a tensor boundary
    check(37, 1000); // arbitrary mid-tensor range

    assert!(source.read_range(n as u64 - 10, &mut vec![0u8; 20]).is_err());
    assert!(source.read_range(n as u64, &mut vec![0u8; 1]).is_err());
}

/// LAYOUT-01 acceptance: after warm-up, dispatching the same expert twice
/// reuses the identical bytes pointers through the cache lease — the fused
/// gate/up weight is a VIEW over record[0 .. gate+up] and the down weight a
/// view over the tail, so a host-cache hit performs zero record-sized
/// allocation or copy.
#[test]
fn fused_gate_up_is_zero_copy_view_over_lease() {
    let fx = stacked_fixture();
    let store = make_store(&fx);
    let shared = StreamShared::new(store, 8 * fx.rec_bytes, 0).expect("shared");
    let stream = shared.layer(0);

    let gate_per = q8_0_bytes(FF * HIDDEN);
    let up_per = q8_0_bytes(FF * HIDDEN);

    let (pair1, down1) = stream.expert_weights(2).expect("dispatch 1 (cold)");
    let (pair2, down2) = stream.expert_weights(2).expect("dispatch 2 (hit)");

    // The fused view path is taken for uniform Q8_0 halves.
    let FfnPair::Fused(Weight::Quant(q1)) = &pair1 else {
        panic!("expected a fused quant view, got {pair1:?}")
    };
    let FfnPair::Fused(Weight::Quant(q2)) = &pair2 else {
        panic!("expected a fused quant view, got {pair2:?}")
    };
    assert_eq!(q1.shape(), &[2 * FF, HIDDEN]);
    assert_eq!(q1.nbytes(), gate_per + up_per);
    // The view reads exactly the record's gate||up prefix.
    assert_eq!(q1.bytes(), &fx.records[0][2][..gate_per + up_per]);

    // Same lease, same pointers: dispatch 2 copied nothing.
    assert_eq!(
        q1.bytes().as_ptr(),
        q2.bytes().as_ptr(),
        "cache-hit dispatch must reuse the leased record bytes"
    );

    // The down projection is a view over the record tail, right behind the
    // fused gate/up region.
    let Weight::Quant(d1) = &down1 else {
        panic!("expected quant down weight, got {down1:?}")
    };
    let Weight::Quant(d2) = &down2 else {
        panic!("expected quant down weight, got {down2:?}")
    };
    assert_eq!(d1.bytes(), &fx.records[0][2][gate_per + up_per..]);
    assert_eq!(
        d1.bytes().as_ptr() as usize,
        q1.bytes().as_ptr() as usize + gate_per + up_per,
        "down view must sit directly behind gate||up in the same lease"
    );
    assert_eq!(d1.bytes().as_ptr(), d2.bytes().as_ptr());

    let st = shared.cache_stats();
    assert_eq!((st.misses, st.hits), (1, 1));
}

/// LAYOUT-01 end-to-end at the layer level, streamed from the file: the
/// streaming forward (routing pre-pass + batch prewarm + fused views) is
/// bitwise-identical to the resident `moe_forward_with_logits` on the same
/// experts, cold and warm. seq=2 exercises the multi-token union prewarm.
#[test]
fn forward_with_logits_matches_resident_moe() {
    let fx = file_fixture();
    let store = make_file_store(&fx);
    let shared = StreamShared::new(store, 8 * fx.rec_bytes, 0).expect("shared");
    let backend = ggml_rs::default_backend();
    let idx = crate::loader::TensorIndex::new(&fx.file);

    let x = Tensor::from_vec(
        (0..2 * HIDDEN)
            .map(|i| (i as f32 - 32.0) / 31.0)
            .collect(),
        vec![2, HIDDEN],
    );
    let logits = Tensor::from_vec(
        (0..2 * N_EXPERTS).map(|i| ((i * 7 + 3) % 13) as f32).collect(),
        vec![2, N_EXPERTS],
    );
    let opts = MoeOptions::default();

    for l in 0..N_LAYERS {
        let gate_stacked = idx
            .take_weight(&format!("blk.{l}.ffn_gate_exps.weight"), &[])
            .expect("gate");
        let up_stacked = idx
            .take_weight(&format!("blk.{l}.ffn_up_exps.weight"), &[])
            .expect("up");
        let down_stacked = idx
            .take_weight(&format!("blk.{l}.ffn_down_exps.weight"), &[])
            .expect("down");
        let gates = crate::qwen3moe::split_stacked_experts(gate_stacked, N_EXPERTS).unwrap();
        let ups = crate::qwen3moe::split_stacked_experts(up_stacked, N_EXPERTS).unwrap();
        let downs = crate::qwen3moe::split_stacked_experts(down_stacked, N_EXPERTS).unwrap();
        let pairs: Vec<FfnPair> = gates
            .into_iter()
            .zip(ups)
            .map(|(g, u)| FfnPair::from_halves(g, u))
            .collect();
        let moe = crate::moe::MoeFfn {
            // Unused by *_with_logits; only the experts matter here.
            router: Weight::Dense(Tensor::from_vec(vec![0.0], vec![1, 1])),
            gate_up_experts: pairs,
            down_experts: downs,
            top_k: 2,
            stream: None,
            #[cfg(feature = "cuda")]
            gpu_plan: std::sync::OnceLock::new(),
        };

        let resident =
            crate::moe::moe_forward_with_logits(&*backend, &x, &moe, &logits, &opts);
        let stream = shared.layer(l as u32);
        let cold = stream.forward_with_logits(&*backend, &x, &logits, 2, &opts);
        let warm = stream.forward_with_logits(&*backend, &x, &logits, 2, &opts);

        assert_f32_bitwise(
            resident.to_host().data(),
            cold.to_host().data(),
            &format!("layer {l}: cold streaming vs resident"),
        );
        assert_f32_bitwise(
            resident.to_host().data(),
            warm.to_host().data(),
            &format!("layer {l}: warm streaming vs resident"),
        );
    }
    assert!(shared.error().is_none());
}

/* ---- Wave C: batch admission ------------------------------------------- */

/// Wave C: `fetch_many` fills N records byte-identical to the per-record
/// path; an empty batch is fine.
#[test]
fn fetch_many_matches_records() {
    let fx = file_fixture();
    let store = make_file_store(&fx);

    let keys: Vec<(u32, u32)> = (0..N_LAYERS as u32)
        .flat_map(|l| (0..N_EXPERTS as u32).map(move |e| (l, e)))
        .collect();
    let mut bufs: Vec<Vec<u8>> = (0..keys.len()).map(|_| vec![0u8; fx.rec_bytes]).collect();
    store.fetch_many(&keys, &mut bufs).expect("fetch_many");
    for ((l, e), buf) in keys.iter().zip(&bufs) {
        assert_eq!(
            buf,
            &fx.records[*l as usize][*e as usize],
            "batched record ({l}, {e}) mismatch"
        );
    }
    store.fetch_many(&[], &mut []).expect("empty batch");
}

/// Wave C: `fetch_many` validates every key and buffer before any I/O and
/// names the bad entry.
#[test]
fn fetch_many_validates_before_reading() {
    let fx = file_fixture();
    let store = make_file_store(&fx);

    let mut bufs: Vec<Vec<u8>> = (0..2).map(|_| vec![0u8; fx.rec_bytes]).collect();
    // Out-of-range key at position 1.
    let err = store
        .fetch_many(&[(0, 0), (N_LAYERS as u32, 0)], &mut bufs)
        .unwrap_err();
    assert!(err.to_string().contains("entry 1"), "{err}");
    // Wrong buffer size at position 1.
    bufs[1] = vec![0u8; fx.rec_bytes - 1];
    let err = store.fetch_many(&[(0, 0), (0, 1)], &mut bufs).unwrap_err();
    assert!(err.to_string().contains("entry 1"), "{err}");
    // Key/buffer count mismatch.
    let err = store.fetch_many(&[(0, 0)], &mut bufs).unwrap_err();
    assert!(err.to_string().contains("1 keys but 2 buffers"), "{err}");
}

/// Wave C acceptance: a layer's cold top-k prewarm reads each missing
/// expert once, a warm re-dispatch reads nothing, and the result is the
/// same either way.
#[test]
fn prewarm_reads_each_cold_expert_once() {
    let fx = file_fixture();
    let store = make_file_store(&fx);
    let shared = StreamShared::new(store, 8 * fx.rec_bytes, 0).expect("shared");
    let backend = ggml_rs::default_backend();

    let x = Tensor::from_vec(
        (0..2 * HIDDEN)
            .map(|i| (i as f32 - 32.0) / 31.0)
            .collect(),
        vec![2, HIDDEN],
    );
    // top-2 over 4 experts, two tokens: distinct demand is 3 experts.
    let logits = Tensor::from_vec(
        (0..2 * N_EXPERTS).map(|i| ((i * 7 + 3) % 13) as f32).collect(),
        vec![2, N_EXPERTS],
    );
    let opts = MoeOptions::default();

    let stream0 = shared.layer(0);
    let cold = stream0.forward_with_logits(&*backend, &x, &logits, 2, &opts);
    assert_eq!((cold.dim(0), cold.dim(1)), (2, HIDDEN));
    assert_eq!(shared.cache_stats().misses, 3, "one read per distinct cold expert");

    // Warm: everything resident, no read at all.
    let warm = stream0.forward_with_logits(&*backend, &x, &logits, 2, &opts);
    assert_eq!(shared.cache_stats().misses, 3, "a warm dispatch reads nothing");
    assert_f32_bitwise(
        cold.to_host().data(),
        warm.to_host().data(),
        "warm dispatch matches cold",
    );

    // A second layer's cold batch reads its own three.
    let stream1 = shared.layer(1);
    let _ = stream1.forward_with_logits(&*backend, &x, &logits, 2, &opts);
    assert!(shared.error().is_none());
    let st = shared.cache_stats();
    assert_eq!(st.misses, 6);
    assert!(st.hits > 0);
}

// VENDORED-LOCAL: glm5next's first blocks are dense, so MoE layer 0 is not block 0.

/// Leading dense blocks in the offset fixture. glm5next ships 3.
const N_DENSE: usize = 3;

/// The same stacked Q8_0 expert tensors as [`stacked_fixture`], but every expert
/// tensor is named `blk.{l + N_DENSE}` — the layout glm5next has, where
/// `leading_dense_block_count` blocks carry a dense FFN and no expert tensors at
/// all. `records` stays indexed MoE-relative, which is what the store addresses.
fn dense_lead_fixture() -> Fixture {
    let (mut metadata, tensors, records, rec_bytes) = fixture_tensors();
    metadata.insert(
        "general.architecture".to_string(),
        Value::String("glm5next".to_string()),
    );
    metadata.insert("glm5next.expert_count".to_string(), Value::U32(N_EXPERTS as u32));
    metadata.insert(
        "glm5next.leading_dense_block_count".to_string(),
        Value::U32(N_DENSE as u32),
    );

    let shifted: Vec<(TensorInfo, Vec<u8>)> = tensors
        .into_iter()
        .map(|(mut info, bytes)| {
            let rest = info.name.strip_prefix("blk.").expect("blk-prefixed name");
            let (l, tail) = rest.split_once('.').expect("blk.<n>.<tail>");
            let l: usize = l.parse().expect("layer index");
            info.name = format!("blk.{}.{tail}", l + N_DENSE);
            (info, bytes)
        })
        .collect();

    let raw = gguf::reader::write_to_vec(&metadata, &shifted, 32).expect("write gguf");
    let probe = GgufFile::from_bytes(raw.clone()).expect("parse gguf");
    let data_start = probe.tensor_data_start() as usize;
    let source = Arc::new(VecSource { data: raw[data_start..].to_vec() });
    let file = GgufFile::from_bytes_with_source(raw, source).expect("parse gguf with source");

    Fixture { file, records, rec_bytes }
}

/// The unoffset resolve probes `blk.0.ffn_gate_exps.weight`. A model with
/// leading dense blocks has no such tensor, so it must fail rather than
/// half-resolve — this is the shape glm5next hit before `resolve_range`.
#[test]
fn resolve_rejects_leading_dense_blocks() {
    let fx = dense_lead_fixture();
    assert!(ExpertLayout::resolve(&fx.file, N_LAYERS, N_EXPERTS).is_err());
}

/// With the offset, the layout resolves and every record still reads back
/// byte-identical to what the fixture wrote.
#[test]
fn resolve_range_streams_experts_after_the_dense_blocks() {
    let fx = dense_lead_fixture();
    let layout = ExpertLayout::resolve_range(&fx.file, N_DENSE, N_LAYERS, N_EXPERTS)
        .expect("resolve offset layout");
    assert_eq!(layout.first_layer(), N_DENSE as u32);
    assert_eq!(layout.record_bytes(), fx.rec_bytes);

    let store = GgufExpertStore::new(fx.file.clone(), layout).expect("store");
    // shape() counts MoE layers, not the model's depth.
    assert_eq!(store.shape(), (N_LAYERS as u32, N_EXPERTS as u32));

    // Keys are MoE-relative: (0, e) is block N_DENSE's expert e.
    let mut got = vec![0u8; fx.rec_bytes];
    for l in 0..N_LAYERS {
        for e in 0..N_EXPERTS {
            store.fetch(l as u32, e as u32, &mut got).expect("fetch record");
            assert_eq!(got, fx.records[l][e], "record ({l}, {e}) mismatch");
        }
    }
}
