//! Checks against the official Pixal3D pipeline's own intermediates (dumped by a
//! reference run into P3D_DIR): each stage given the reference's exact inputs.
//!   cargo test --release --features flash-attn --lib model3d::verify -- --ignored --nocapture
use super::decoder::SparseDecoder;
use super::sparse::Level;
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(std::env::var("P3D_DIR").unwrap_or_else(|_| "E:/p3dref/cmp4/ref".into()))
}

fn models() -> PathBuf {
    PathBuf::from(std::env::var("MODELS").unwrap_or_else(|_| "E:/models".into()))
}

fn f32s(name: &str) -> Vec<f32> {
    std::fs::read(dir().join(name)).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn coords(name: &str) -> Vec<[i32; 3]> {
    let v: Vec<i32> = std::fs::read(dir().join(name)).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Relative RMS error and correlation of `mine` against `reference`.
fn compare(label: &str, reference: &[f32], mine: &[f32]) -> f64 {
    let n = reference.len() as f64;
    let (mut d2, mut r2) = (0f64, 0f64);
    let (ma, mb) = (reference.iter().map(|&v| v as f64).sum::<f64>() / n, mine.iter().map(|&v| v as f64).sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
    for (&a, &b) in reference.iter().zip(mine) {
        let (a, b) = (a as f64, b as f64);
        d2 += (a - b) * (a - b);
        r2 += a * a;
        sab += (a - ma) * (b - mb);
        saa += (a - ma) * (a - ma);
        sbb += (b - mb) * (b - mb);
    }
    let rel = (d2 / r2.max(1e-30)).sqrt();
    println!("{label}: rel RMS {rel:.3e}, corr {:.6}", sab / (saa * sbb).sqrt().max(1e-30));
    rel
}

/// The reference's final shape and texture latents, decoded here: the same voxels, and values within half-precision noise.
#[test]
#[ignore]
fn decoders_match_the_reference_on_its_latents() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let ckpts = models().join("Pixal3D/ckpts");
    let hr = coords("coords_hr.i32");
    let n = hr.len();
    let level = Level::new(hr, 64, &dev)?;
    let shape = Tensor::from_vec(f32s("shape_hr_final.bin"), (n, 32), &dev)?;
    let decoder = SparseDecoder::load(&ckpts.join("shape_dec_next_dc_f16c32_fp16"), &dev)?;
    let t0 = std::time::Instant::now();
    let decoded = decoder.forward(level.clone(), &shape, None, |_, _| {})?;
    println!("shape decoded in {:.1}s: {} voxels at {}", t0.elapsed().as_secs_f64(), decoded.level.len(), decoded.level.res);
    drop(decoder);
    let ref_coords = coords("shape_voxels_coords.i32");
    println!("reference: {} voxels", ref_coords.len());
    // Values of the voxels both have, by coordinate.
    let at: HashMap<[i32; 3], usize> = ref_coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let common: Vec<(usize, usize)> = decoded.level.coords.iter().enumerate().filter_map(|(i, c)| at.get(c).map(|&j| (i, j))).collect();
    println!("common voxels: {} ({:.3}% of ours, {:.3}% of theirs)", common.len(), 100. * common.len() as f64 / decoded.level.len() as f64, 100. * common.len() as f64 / ref_coords.len() as f64);
    let mine: Vec<f32> = decoded.feats.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let theirs = &f32s("shape_voxels.bin");
    let mine = &mine;
    for (k, label) in [(0..3, "dual vertex logits"), (3..6, "edge flags"), (6..7, "split weight")] {
        let a: Vec<f32> = common.iter().flat_map(|&(_, j)| k.clone().map(move |c| theirs[j * 7 + c])).collect();
        let b: Vec<f32> = common.iter().flat_map(|&(i, _)| k.clone().map(move |c| mine[i * 7 + c])).collect();
        compare(label, &a, &b);
    }
    let flags_agree = common.iter().filter(|&&(i, j)| (3..6).all(|c| (mine[i * 7 + c] > 0.) == (theirs[j * 7 + c] > 0.))).count();
    println!("edge flags agree on {:.3}% of common voxels", 100. * flags_agree as f64 / common.len() as f64);
    // The texture decoder, following our choice of children.
    let tex = Tensor::from_vec(f32s("tex_hr_final.bin"), (n, 32), &dev)?;
    let tex_decoder = SparseDecoder::load(&ckpts.join("tex_dec_next_dc_f16c32_fp16"), &dev)?;
    let attrs = tex_decoder.forward(level, &tex, Some(&decoded.subdivisions), |_, _| {})?;
    let mine: Vec<f32> = attrs.feats.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let theirs = &f32s("tex_voxels_raw.bin");
    let mine = &mine;
    let a: Vec<f32> = common.iter().flat_map(|&(_, j)| (0..6).map(move |c| theirs[j * 6 + c])).collect();
    let b: Vec<f32> = common.iter().flat_map(|&(i, _)| (0..6).map(move |c| mine[i * 6 + c])).collect();
    compare("texture attributes", &a, &b);
    Ok(())
}

/// One step of the 512 shape flow model (sparse voxels, RoPE by their coordinates) on the reference's inputs.
#[test]
#[ignore]
fn sparse_flow_step() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let c = coords("coords32.i32");
    let n = c.len();
    let dit = super::dit::Dit::load(&models().join("Pixal3D/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16"), &dev)?;
    let x = Tensor::from_vec(f32s("shape_lr.bin"), (n, 32), &dev)?;
    let g = Tensor::from_vec(f32s("global512.bin"), (5, 1024), &dev)?;
    let p = Tensor::from_vec(f32s("proj_shape512.bin"), (n, 2048), &dev)?;
    let rope = super::dit::rope_tables(&c, 128, &dev)?;
    let pos = dit.forward(&x, 1000., &rope, &dit.context(Some(&g), Some(&p), 5, &dev)?)?;
    let neg = dit.forward(&x, 1000., &rope, &dit.context(None, None, 5, &dev)?)?;
    let v = |t: &Tensor| -> Result<Vec<f32>> { t.to_dtype(DType::F32)?.flatten_all()?.to_vec1() };
    compare("sparse step, conditional", &f32s("sparse_pos.bin"), &v(&pos)?);
    compare("sparse step, unconditional", &f32s("sparse_neg.bin"), &v(&neg)?);
    Ok(())
}

/// The meshing after the decoders, from a run's dumped voxels (NROB_DUMP: voxels.i32,
/// shape_voxels.bin, tex_voxels.bin): remeshed (at NROB_REMESH, default 1024),
/// simplified and baked, with each step's time and edges, written as model.glb
/// (textured) and colours.glb.
#[test]
#[ignore]
fn meshing_passes() -> Result<()> {
    let dump = PathBuf::from(std::env::var("NROB_DUMP").unwrap_or_else(|_| "E:/p3dref/cmp4/nrob".into()));
    let read = |n: &str| -> Vec<f32> { std::fs::read(dump.join(n)).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let v: Vec<i32> = std::fs::read(dump.join("voxels.i32")).unwrap().chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let c: Vec<[i32; 3]> = v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let attrs = read("tex_voxels.bin");
    let raw = super::mesh::dual_grid(&c, &read("shape_voxels.bin"), 1024);
    let edges = |m: &super::mesh::Mesh| {
        let mut count: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &m.triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        (count.values().filter(|&&n| n == 1).count(), count.values().filter(|&&n| n > 2).count())
    };
    let volume = |m: &super::mesh::Mesh| -> f64 {
        m.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| m.positions[i as usize].map(f64::from));
                (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.
            })
            .sum()
    };
    let res = std::env::var("NROB_REMESH").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let t = std::time::Instant::now();
    let surface = super::remesh::Surface::new(&raw, res);
    let mut m = super::remesh::remesh(&surface);
    println!("remesh {res}: {} triangles, {} vertices, volume {:.5} in {:.1}s; open/non-manifold {:?}", m.triangles.len(), m.positions.len(), volume(&m), t.elapsed().as_secs_f64(), edges(&m));
    let t = std::time::Instant::now();
    super::simplify(&mut m, 200_000);
    println!("simplified to {} triangles in {:.1}s, volume {:.5}; open/non-manifold {:?}", m.triangles.len(), t.elapsed().as_secs_f64(), volume(&m), edges(&m));
    let voxels = super::mesh::Voxels::new(&c, &attrs, 1024);
    let t = std::time::Instant::now();
    let baked = super::bake::bake(&m, &surface, &voxels, 2048)?;
    println!(
        "baked in {:.1}s: {} charts, {} faces coloured per vertex, {} vertices, {:.0}% of the atlas covered, textures {} + {} KB",
        t.elapsed().as_secs_f64(),
        baked.charts,
        baked.painted,
        baked.mesh.positions.len(),
        baked.coverage * 100.,
        baked.textures.base_color_png.len() / 1024,
        baked.textures.metallic_roughness_png.len() / 1024
    );
    std::fs::write(dump.join("model.glb"), super::glb::write(&baked.mesh, Some(&baked.textures)))?;
    m.color_from_voxels(&voxels);
    std::fs::write(dump.join("colours.glb"), super::glb::write(&m, None))?;
    Ok(())
}
