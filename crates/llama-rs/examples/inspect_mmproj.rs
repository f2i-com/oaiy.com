//! Validate an mmproj.gguf (vision-tower companion file). Prints the parsed
//! `MmProjConfig`, runs `MmProj::from_gguf` to load + shape-check every
//! tensor against the config, and reports any mismatch in human-readable
//! form. Use this before wiring an mmproj into the LM input stream — it'll
//! surface tensor-naming or dimension issues immediately rather than at
//! forward time as a NaN.
//!
//! Usage:
//!   cargo run --release -p llama-rs --example inspect_mmproj -- mmproj.gguf
//!   cargo run --release -p llama-rs --example inspect_mmproj -- mmproj.gguf --forward
//!
//! `--forward` synthesises a solid-grey 896×896 image, runs the full ViT +
//! projector pipeline, and prints the output shape — sanity-checks the
//! forward path without needing a real photo.

use std::env;

use ggml_rs::default_backend;
use gguf::GgufFile;
use llama_rs::{preprocess_image_bytes, MmProj, MmProjConfig, Projector, VisionConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: inspect_mmproj <mmproj.gguf> [--forward]")?;
    let do_forward = args.any(|a| a == "--forward" || a == "-f");

    let g = GgufFile::open(&path)?;

    // Print raw config first — useful even if the loader fails on shape
    // validation since you'd want to know what the file *claims* to be.
    let cfg = MmProjConfig::from_gguf(&g)?;
    println!("== MmProjConfig ==");
    println!("  image_size:     {}", cfg.image_size);
    println!("  patch_size:     {}", cfg.patch_size);
    println!("  embedding_dim:  {}", cfg.embedding_dim);
    println!("  n_layers:       {}", cfg.n_layers);
    println!("  n_heads:        {}", cfg.n_heads);
    println!("  head_dim:       {}", cfg.head_dim);
    println!("  ff_dim:         {}", cfg.ff_dim);
    println!("  layer_norm_eps: {}", cfg.layer_norm_eps);
    println!("  projector:      {:?}", cfg.projector);
    println!("  mean:           {:?}", cfg.mean);
    println!("  std:            {:?}", cfg.std);
    println!("  n_patches:      {}", cfg.n_patches());
    println!("  n_soft_tokens:  {} (after projector pooling)", cfg.n_soft_tokens());
    println!();

    println!("Loading + shape-validating tensors on CPU backend ...");
    let backend = default_backend();
    let mm = MmProj::from_gguf(&g, backend.clone())?;
    println!("  ✓ all tensors load and pass shape validation");
    let (kind_label, pre_ln) = match &mm {
        MmProj::SigLip(m) => {
            let p = match &m.projector {
                Projector::Gemma(_) => "SigLIP tower + Gemma (avg-pool 4×4 + RMSNorm + linear)",
                Projector::Mlp(_)   => "SigLIP tower + Mlp (linear → GeLU → linear)",
            };
            (p, m.pre_ln.is_some())
        }
        MmProj::Gemma4V(_) => ("Gemma 4 V tower (RMSNorm + SwiGLU + sandwich norms) + single linear projector", false),
        MmProj::Qwen3Vl(_) => ("Qwen3-VL tower (LN + fused QKV + GeLU MLP + post-LN) + 2x2 spatial-merge MLP head", false),
    };
    println!("  vision blocks:  {}", mm.n_blocks());
    println!("  flavour:        {kind_label}");
    println!("  pre-LN present: {pre_ln}");
    println!();

    if do_forward {
        println!("== Forward pass on synthetic mid-grey image ==");
        // Build a solid 256×256 mid-grey PNG in memory; preprocess will resize
        // it to the model's expected input (e.g. 896 for SigLIP-Gemma).
        let mut img = image::RgbImage::new(256, 256);
        for px in img.pixels_mut() { *px = image::Rgb([128, 128, 128]); }
        let mut buf = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)?;

        // Build a VisionConfig from the mmproj config so the normalisation
        // constants come from the GGUF (not the SIGLIP_GEMMA preset, which
        // might disagree with what the converter wrote out).
        let vc = VisionConfig {
            image_size: cfg.image_size,
            patch_size: cfg.patch_size,
            mean: cfg.mean,
            std:  cfg.std,
        };
        let image = preprocess_image_bytes(&buf, &vc)?;
        println!("  preprocessed image shape: {:?}", image.shape());

        let t0 = std::time::Instant::now();
        let soft_tokens = mm.forward(&image)?;
        let dt = t0.elapsed();
        println!("  soft-token shape:         {:?}", soft_tokens.shape());
        println!("  forward elapsed:          {:.2?}", dt);
    }

    Ok(())
}
