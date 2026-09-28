//! Per-operation cost on a device: tiny tensor ops, a talker-sized matmul,
//! one talker step and one predictor step.
//!
//! cargo run --release -p oaiy-tts --features flash-attn --example ops -- --device 1
use candle_core::{DType, Tensor};
use oaiy_tts::model::Cache;
use oaiy_tts::talker::Talker;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut device, mut model) = (0usize, PathBuf::from("E:/models/Qwen3-TTS-12Hz-0.6B-Base"));
    while let Some(a) = args.next() {
        match a.as_str() {
            "--device" => device = args.next().ok_or("--device N")?.parse()?,
            "--model" => model = args.next().ok_or("--model DIR")?.into(),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let dev = oaiy_tts::cuda(device)?;
    let x = Tensor::zeros((1, 1, 1024), DType::BF16, &dev)?;
    let w = Tensor::zeros((3072, 1024), DType::BF16, &dev)?;
    let time = |name: &str, n: usize, f: &mut dyn FnMut() -> candle_core::Result<()>| -> candle_core::Result<()> {
        for _ in 0..10 {
            f()?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..n {
            f()?;
        }
        let queued = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        dev.synchronize()?;
        let done = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        println!("{name}: {queued:.1} us to queue, {done:.1} us done");
        Ok(())
    };
    time("add (1x1024 bf16)", 2000, &mut || (&x + &x).map(|_| ()))?;
    time("matmul 1x1024 @ 1024x3072", 2000, &mut || x.broadcast_matmul(&w.t()?).map(|_| ()))?;
    time("  same, 100 in a row", 100, &mut || x.broadcast_matmul(&w.t()?).map(|_| ()))?;
    let x2 = x.reshape((1, 1024))?;
    time("  2-D x @ w.t()", 100, &mut || x2.matmul(&w.t()?).map(|_| ()))?;
    let wt = w.t()?.contiguous()?;
    time("  2-D x @ w stored transposed", 100, &mut || x2.matmul(&wt).map(|_| ()))?;
    let xc = x2.t()?.contiguous()?;
    time("  w @ x (column)", 100, &mut || w.matmul(&xc).map(|_| ()))?;
    let x8 = Tensor::zeros((8, 1024), DType::BF16, &dev)?;
    time("  8 rows @ w.t()", 100, &mut || x8.matmul(&w.t()?).map(|_| ()))?;
    let (xf, wf) = (x2.to_dtype(DType::F32)?, w.to_dtype(DType::F32)?);
    time("  F32 x @ w.t()", 100, &mut || xf.matmul(&wf.t()?).map(|_| ()))?;
    let (xh, wh) = (x2.to_dtype(DType::F16)?, w.to_dtype(DType::F16)?);
    time("  F16 x @ w.t()", 100, &mut || xh.matmul(&wh.t()?).map(|_| ()))?;
    let small = Tensor::zeros((64, 1024), DType::BF16, &dev)?;
    time("  1x1024 @ 1024x64 (tiny weight)", 100, &mut || x2.matmul(&small.t()?).map(|_| ()))?;
    let big = Tensor::zeros((3072 * 4, 1024), DType::BF16, &dev)?;
    time("  1x1024 @ 1024x12288 (4x the weight)", 100, &mut || x2.matmul(&big.t()?).map(|_| ()))?;
    let x64 = Tensor::zeros((64, 1024), DType::BF16, &dev)?;
    time("  64 rows @ w.t()", 100, &mut || x64.matmul(&w.t()?).map(|_| ()))?;
    time("  sqr (1x1024)", 100, &mut || x2.sqr().map(|_| ()))?;
    let norm_w = Tensor::ones(1024, DType::BF16, &dev)?;
    time("  rms_norm (1x1024)", 100, &mut || candle_nn::ops::rms_norm(&x, &norm_w, 1e-6).map(|_| ()))?;
    let qkv = Tensor::zeros((1, 1, 4096), DType::BF16, &dev)?;
    time("  narrow+reshape+contiguous (q from qkv)", 100, &mut || qkv.narrow(2, 0, 2048)?.reshape((1, 1, 16, 128)).map(|_| ()))?;
    let q = Tensor::zeros((1, 1, 16, 128), DType::BF16, &dev)?;
    let hw = Tensor::ones(128, DType::BF16, &dev)?;
    time("  rms_norm per head (16x128)", 100, &mut || candle_nn::ops::rms_norm(&q, &hw, 1e-6).map(|_| ()))?;
    let (cos, sin) = (Tensor::zeros((1, 64), DType::BF16, &dev)?, Tensor::zeros((1, 64), DType::BF16, &dev)?);
    time("  rope_thd", 100, &mut || candle_nn::rotary_emb::rope_thd(&q, &cos, &sin).map(|_| ()))?;
    let kc = Tensor::zeros((1, 100, 8, 128), DType::BF16, &dev)?;
    let k1 = Tensor::zeros((1, 1, 8, 128), DType::BF16, &dev)?;
    time("  cat onto a 100-long cache", 100, &mut || Tensor::cat(&[&kc, &k1], 1).map(|_| ()))?;
    #[cfg(feature = "flash-attn")]
    time("  flash_attn 1 query, 100 keys", 100, &mut || candle_flash_attn::flash_attn(&q, &kc, &kc, 0.088, true).map(|_| ()))?;
    let gu = Tensor::zeros((1, 1, 6144), DType::BF16, &dev)?;
    time("  silu(gate) * up from the fused output", 100, &mut || (gu.narrow(2, 0, 3072)?.silu()? * gu.narrow(2, 3072, 3072)?).map(|_| ()))?;
    let table = Tensor::zeros((2048, 1024), DType::BF16, &dev)?;
    let id = Tensor::new(&[5u32], &dev)?;
    time("  index_select one row", 100, &mut || table.index_select(&id, 0).map(|_| ()))?;
    let l = Tensor::zeros((1, 2048), DType::F32, &dev)?;
    time("  arg_sort 2048", 100, &mut || l.arg_sort_last_dim(false).map(|_| ()))?;
    time("to_vec1 of 16 u32", 500, &mut || Tensor::zeros(16, DType::U32, &dev)?.to_vec1::<u32>().map(|_| ()))?;
    time("upload 16 u32", 2000, &mut || Tensor::new(&[0u32; 16], &dev).map(|_| ()))?;
    let talker = Talker::load(&model, &dev)?;
    let dec = talker.decoder();
    let mut cache = Cache::new(dec.layers());
    dec.forward(&Tensor::zeros((1, 100, 1024), DType::BF16, &dev)?, &mut cache)?;
    let one = Tensor::zeros((1, 1, 1024), DType::BF16, &dev)?;
    time("talker step (28 layers, 1 token)", 50, &mut || dec.forward(&one, &mut cache).map(|_| ()))?;
    Ok(())
}
