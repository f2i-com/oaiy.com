use super::sparse::{Conv, Level, Subdivision};
use candle_core::{DType, Device, Result, Tensor};

/// A tiny deterministic generator (no rand dependency in tests).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

#[test]
fn a_sparse_conv_is_a_zero_padded_3d_convolution_over_the_voxels_there_are() -> Result<()> {
    let dev = Device::Cpu;
    let res = 5;
    // Every other voxel of a 5³ grid, and a few more.
    let coords: Vec<[i32; 3]> = (0..res as i32).flat_map(|x| (0..res as i32).flat_map(move |y| (0..res as i32).map(move |z| [x, y, z]))).filter(|c| (c[0] + c[1] + c[2]) % 2 == 0 || c[0] == 2).collect();
    let (ci, co) = (3, 4);
    let n = coords.len();
    let level = Level::new(coords.clone(), res, &dev)?;
    let feats = values(n * ci, 1);
    let weight = values(co * 27 * ci, 2);
    let bias = values(co, 3);
    let x = Tensor::from_vec(feats.clone(), (n, ci), &dev)?;
    let w = Tensor::from_vec(weight.clone(), (co, 3, 3, 3, ci), &dev)?;
    let conv = Conv::sparse(&w, Some(Tensor::from_vec(bias.clone(), co, &dev)?), DType::F32)?;
    let y: Vec<Vec<f32>> = conv.forward(&level, &x)?.to_vec2()?;
    let at = |c: [i32; 3]| coords.iter().position(|&d| d == c);
    for (i, c) in coords.iter().enumerate() {
        for o in 0..co {
            let mut want = bias[o];
            for kd in 0..3 {
                for kh in 0..3 {
                    for kw in 0..3 {
                        if let Some(j) = at([c[0] + kd as i32 - 1, c[1] + kh as i32 - 1, c[2] + kw as i32 - 1]) {
                            for k in 0..ci {
                                want += weight[(((o * 3 + kd) * 3 + kh) * 3 + kw) * ci + k] * feats[j * ci + k];
                            }
                        }
                    }
                }
            }
            assert!((y[i][o] - want).abs() < 1e-4, "voxel {c:?} channel {o}: {} vs {want}", y[i][o]);
        }
    }
    Ok(())
}

#[test]
fn channel_to_space_gives_each_kept_child_its_slice_of_the_parent() -> Result<()> {
    let dev = Device::Cpu;
    let level = Level::new(vec![[0, 0, 0], [1, 0, 1]], 2, &dev)?;
    // Children 0 and 7 of the first voxel, child 5 of the second.
    let mut logits = vec![-1f32; 16];
    logits[0] = 1.;
    logits[7] = 1.;
    logits[8 + 5] = 1.;
    let sub = Subdivision::from_logits(&Tensor::from_vec(logits, (2, 8), &dev)?)?;
    assert_eq!(sub.coords(&level), vec![[0, 0, 0], [1, 1, 1], [3, 0, 3]]);
    let x = Tensor::arange(0f32, 32., &dev)?.reshape((2, 16))?;
    let children: Vec<Vec<f32>> = sub.channel_to_space(&x)?.to_vec2()?;
    assert_eq!(children, vec![vec![0., 1.], vec![14., 15.], vec![26., 27.]]);
    Ok(())
}

fn gpu() -> Device {
    Device::cuda_if_available(0).unwrap()
}

fn models() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("MODELS").unwrap_or_else(|_| "E:/models".into()))
}

/// Every Pixal3D part loads, and runs on something small.   cargo test --release --features flash-attn --lib model3d -- --ignored
#[test]
#[ignore]
fn every_part_loads_and_runs() -> Result<()> {
    let dev = gpu();
    let ckpts = models().join("Pixal3D/ckpts");
    let t0 = std::time::Instant::now();
    let dit = super::dit::Dit::load(&ckpts.join("ss_flow_img_dit_1_3B_64_bf16"), &dev)?;
    println!("ss flow loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let coords: Vec<[i32; 3]> = (0..4).map(|i| [i, 0, 0]).collect();
    let rope = super::dit::rope_tables(&coords, 128, &dev)?;
    let ctx = dit.context(Some(&Tensor::zeros((5, 1024), DType::F32, &dev)?), Some(&Tensor::zeros((4, 1024), DType::F32, &dev)?), 5, &dev)?;
    let v = dit.forward(&Tensor::zeros((4, 8), DType::F32, &dev)?, 500., &rope, &ctx)?;
    println!("ss flow out {:?}", v.dims());
    drop(dit);
    let dec = super::decoder::StructureDecoder::load(&ckpts.join("ss_dec_conv3d_16l8_fp16"), &dev)?;
    let (occ, res) = dec.forward(&Tensor::zeros((16 * 16 * 16, 8), DType::F32, &dev)?)?;
    println!("structure decoder {:?} at {res}", occ.dims());
    let shape = super::decoder::SparseDecoder::load(&ckpts.join("shape_dec_next_dc_f16c32_fp16"), &dev)?;
    let level = super::sparse::Level::new(vec![[10, 10, 10], [10, 10, 11]], 64, &dev)?;
    let out = shape.forward(level, &Tensor::zeros((2, 32), DType::F32, &dev)?, None, |_, _| {})?;
    println!("shape decoder: {} voxels at {}", out.level.len(), out.level.res);
    let dino = super::dinov3::Dinov3::load(&models().join("dinov3-vitl16"), &dev)?;
    let tokens = dino.forward(&Tensor::zeros((3, 64, 64), DType::F32, &dev)?)?;
    println!("dinov3 {:?}", tokens.dims());
    let naf = super::naf::Naf::load(&models().join("NAF/naf_release.pth"), &dev)?;
    let hr = naf.at_pixels(&Tensor::zeros((3, 512, 512), DType::F32, &dev)?, &Tensor::zeros((1024, 32, 32), DType::F32, &dev)?, 512, 512, &[0, 1, 5000])?;
    println!("naf {:?}", hr.dims());
    Ok(())
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn write_f32(path: &std::path::Path, t: &Tensor) {
    let v: Vec<f32> = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

/// One step of the structure flow model on the reference's dumped inputs (P3D_DIR).
#[test]
#[ignore]
fn structure_flow_step() -> Result<()> {
    let dir = std::path::PathBuf::from(std::env::var("P3D_DIR").unwrap_or_else(|_| "E:/p3dref/cmp4/ref".into()));
    let dev = gpu();
    let dit = super::dit::Dit::load(&models().join("Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16"), &dev)?;
    let x = Tensor::from_vec(read_f32(&dir.join("ss.bin")), (8, 4096), &dev)?.t()?.contiguous()?;
    let g = Tensor::from_vec(read_f32(&dir.join("global512.bin")), (5, 1024), &dev)?;
    let p = Tensor::from_vec(read_f32(&dir.join("proj_ss.bin")), (4096, 1024), &dev)?;
    let coords = super::sparse::Level::dense(16, &dev)?.coords.clone();
    let rope = super::dit::rope_tables(&coords, 128, &dev)?;
    let pos = dit.context(Some(&g), Some(&p), 5, &dev)?;
    let neg = dit.context(None, None, 5, &dev)?;
    write_f32(&dir.join("nrob_step_pos.bin"), &dit.forward(&x, 1000., &rope, &pos)?);
    write_f32(&dir.join("nrob_step_neg.bin"), &dit.forward(&x, 1000., &rope, &neg)?);
    write_f32(&dir.join("nrob_step_mid.bin"), &dit.forward(&(&x * 0.5)?, 400., &rope, &pos)?);
    Ok(())
}

/// A box from `lo` to `hi` as 12 triangles facing out.
fn cuboid(lo: [f32; 3], hi: [f32; 3]) -> super::mesh::Mesh {
    let positions = (0..8).map(|i| [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }]).collect();
    let quads: [[u32; 4]; 6] = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    let triangles = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).collect();
    super::mesh::Mesh { positions, triangles, colors: None, metal_rough: None }
}

fn volume(m: &super::mesh::Mesh) -> f32 {
    m.triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| m.positions[i as usize]);
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.
        })
        .sum()
}

#[test]
fn a_closed_surface_remeshes_to_its_outer_sheet_and_an_open_one_to_both() {
    let cube = cuboid([-0.25; 3], [0.25; 3]);
    assert!((volume(&cube) - 0.125).abs() < 1e-6, "the box faces out");
    let out = super::remesh::remesh(&super::remesh::Surface::new(&cube, 64));
    // One voxel out: (0.5 + 2 · 67/64/64)³, a little less for the rounded edges; with
    // the inner sheet too it would be about a third of that.
    let v = volume(&out);
    assert!((0.13..0.152).contains(&v), "volume {v}");
    assert!(out.triangles.len() > 1000);
    // A square sheet has no inside: both of its sides stay.
    let sheet = super::mesh::Mesh { positions: vec![[-0.25, -0.25, 0.], [0.25, -0.25, 0.], [0.25, 0.25, 0.], [-0.25, 0.25, 0.]], triangles: vec![[0, 1, 2], [0, 2, 3]], colors: None, metal_rough: None };
    let out = super::remesh::remesh(&super::remesh::Surface::new(&sheet, 64));
    let facing = |sign: f32| {
        out.triangles
            .iter()
            .filter(|t| {
                let [a, b, c] = t.map(|i| out.positions[i as usize]);
                let n = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
                n * sign > 0. && a[2] * sign > 0.
            })
            .count()
    };
    assert!(facing(1.) > 100 && facing(-1.) > 100, "{} up, {} down", facing(1.), facing(-1.));
}

#[test]
fn a_baked_box_has_its_faces_colours_in_its_texture() -> Result<()> {
    // The box's surface voxels at 64: red where x > 0, blue elsewhere, half metallic.
    let res = 64usize;
    let cube = cuboid([-0.25; 3], [0.25; 3]);
    let mut coords = Vec::new();
    let mut attrs = Vec::new();
    for x in 0..res as i32 {
        for y in 0..res as i32 {
            for z in 0..res as i32 {
                let c = [x, y, z].map(|v| (v as f32 + 0.5) / res as f32 - 0.5);
                if c.iter().all(|v| v.abs() < 0.27) && c.iter().any(|v| v.abs() > 0.23) {
                    coords.push([x, y, z]);
                    attrs.extend(if c[0] > 0. { [1., 0., 0., 0.5, 0.25, 1.] } else { [0., 0., 1., 0.5, 0.25, 1.] });
                }
            }
        }
    }
    let surface = super::remesh::Surface::new(&cube, res);
    let voxels = super::mesh::Voxels::new(&coords, &attrs, res);
    let baked = super::bake::bake(&cube, &surface, &voxels, 512)?;
    assert_eq!((baked.charts, baked.painted), (6, 0));
    assert!(baked.mesh.colors.is_none());
    assert_eq!(baked.mesh.triangles.len(), 12);
    assert_eq!(baked.textures.uvs.len(), baked.mesh.positions.len());
    assert!(baked.textures.uvs.iter().flatten().all(|v| (0. ..=1.).contains(v)));
    assert!(baked.coverage > 0.3, "coverage {}", baked.coverage);
    assert!(!baked.textures.transparent);
    let base = image::load_from_memory(&baked.textures.base_color_png).map_err(candle_core::Error::wrap)?.to_rgba8();
    let mr = image::load_from_memory(&baked.textures.metallic_roughness_png).map_err(candle_core::Error::wrap)?.to_rgb8();
    assert_eq!(base.dimensions(), (512, 512));
    // Each face's middle, in the texture, is its side's colour.
    for t in &baked.mesh.triangles {
        let p = t.map(|i| baked.mesh.positions[i as usize]);
        let uv = t.map(|i| baked.textures.uvs[i as usize]);
        let m = |k: usize| (uv[0][k] + uv[1][k] + uv[2][k]) / 3. * 512.;
        let x = (p[0][0] + p[1][0] + p[2][0]) / 3.;
        let want = if x > 0.2 {
            [255, 0, 0, 255]
        } else if x < -0.2 {
            [0, 0, 255, 255]
        } else {
            continue;
        };
        assert_eq!(base.get_pixel(m(0) as u32, m(1) as u32).0, want, "face at x {x}");
        assert_eq!(mr.get_pixel(m(0) as u32, m(1) as u32).0, [0, 64, 128]);
    }
    Ok(())
}

#[test]
fn faces_that_never_show_are_coloured_per_vertex() -> Result<()> {
    // A box inside a box: the inner one's faces are hidden from every side.
    let mut mesh = cuboid([-0.25; 3], [0.25; 3]);
    let inner = cuboid([-0.1; 3], [0.1; 3]);
    mesh.positions.extend(&inner.positions);
    mesh.triangles.extend(inner.triangles.iter().map(|t| t.map(|i| i + 8)));
    let coords: Vec<[i32; 3]> = (0..64).flat_map(|x| (0..64).map(move |y| [x, y, 48])).collect();
    let attrs: Vec<f32> = coords.iter().flat_map(|_| [0., 1., 0., 0., 1., 1.]).collect();
    let surface = super::remesh::Surface::new(&mesh, 64);
    let baked = super::bake::bake(&mesh, &surface, &super::mesh::Voxels::new(&coords, &attrs, 64), 256)?;
    assert_eq!((baked.charts, baked.painted), (6, 12));
    let colors = baked.mesh.colors.as_ref().expect("vertex colours");
    // The charts' vertices take the texture as it is; the painted ones point at white.
    let painted: Vec<usize> = baked.mesh.triangles[12..].iter().flatten().map(|&i| i as usize).collect();
    assert!(painted.iter().all(|&i| baked.textures.uvs[i] == baked.textures.uvs[painted[0]]));
    assert!(baked.mesh.triangles[..12].iter().flatten().all(|&i| colors[i as usize] == [1.; 4]));
    let base = image::load_from_memory(&baked.textures.base_color_png).map_err(candle_core::Error::wrap)?.to_rgba8();
    let uv = baked.textures.uvs[painted[0]];
    assert_eq!(base.get_pixel((uv[0] * 256.) as u32, (uv[1] * 256.) as u32).0, [255; 4]);
    Ok(())
}

/// Pictures prepared with BiRefNet and Real-ESRGAN on the GPU, written to
/// E:/p3dref/prepare for a look: the bust (plain background), the bust at 300
/// pixels (enlarged), and the turtle on a busy photo (cut out of it).
/// cargo test --release --features flash-attn --lib model3d::tests::pictures_are_cut_out_and_enlarged -- --ignored --nocapture
#[test]
#[ignore]
fn pictures_are_cut_out_and_enlarged() -> Result<()> {
    let out = std::path::PathBuf::from("E:/p3dref/prepare");
    std::fs::create_dir_all(&out)?;
    let gpu: usize = std::env::var("NROB_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let dev = Device::cuda_if_available(gpu)?;
    let bust = image::open("E:/p3dref/case4/input.png").unwrap().to_rgb8();
    image::imageops::resize(&bust, 300, 300, image::imageops::FilterType::Lanczos3).save(out.join("small.png")).unwrap();
    let turtle = image::open(std::env::var("TURTLE").unwrap_or_else(|_| "C:/Users/User/AppData/Local/Temp/claude/E--repos-bot-computer/fc80cba4-4bf0-4581-8aca-668cb05bf606/scratchpad/pixal3d/repo/assets/images/0_img.png".into())).unwrap().to_rgba8();
    let turtle = image::imageops::resize(&turtle, 900, 900, image::imageops::FilterType::Lanczos3);
    let mut scene = image::imageops::resize(&image::open(std::env::var("SCENE").unwrap_or_else(|_| "C:/Users/User/AppData/Local/Temp/claude/E--repos-bot-computer/fc80cba4-4bf0-4581-8aca-668cb05bf606/scratchpad/pixal3d/repo/assets/app/hdri_city.png".into())).unwrap().to_rgba8(), 1200, 1000, image::imageops::FilterType::Lanczos3);
    image::imageops::overlay(&mut scene, &turtle, 150, 60);
    image::DynamicImage::ImageRgba8(scene).to_rgb8().save(out.join("busy.png")).unwrap();
    let helpers = super::prepare::Helpers { matte: Some(std::path::Path::new("E:/models/BiRefNet")), upscaler: Some(std::path::Path::new("E:/models/Real-ESRGAN/RealESRGAN_x4plus.pth")), dev: &dev };
    for (name, picture) in [("bust", std::path::PathBuf::from("E:/p3dref/case4/input.png")), ("small", out.join("small.png")), ("busy", out.join("busy.png"))] {
        let t = std::time::Instant::now();
        let p = super::prepare::prepare(&picture, &helpers)?;
        println!("{name}: cut out by {}, {}×{}, square {:?}, in {:.2}s", p.matte, p.image.width(), p.image.height(), p.upscaled, t.elapsed().as_secs_f64());
        p.image.save(out.join(format!("{name}-on-black.png"))).unwrap();
        p.cutout.save(out.join(format!("{name}-cutout.png"))).unwrap();
    }
    Ok(())
}
