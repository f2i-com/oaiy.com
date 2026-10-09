//! `vision`'s tests.

use super::*;

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i * 2654435761 + seed * 40503) % 1009;
            (h as f32 - 504.0) / 5040.0
        })
        .collect()
}

fn shape() -> VisionShape {
    VisionShape {
        n_embd: 8,
        n_layer: 2,
        n_head: 2,
        n_ff: 12,
        patch: 2,
        n_merge: 2,
        proj_dim: 6,
        proj_ff: 10,
        swiglu_limit: 10.0,
        eps: 1e-5,
        proj_norm_eps: 1e-5,
    }
}

/// The merger reads its group **channel-major**. This pins the exact index
/// mapping, which a slot/channel transpose would silently break on real
/// weights.
#[test]
fn the_merger_window_is_channel_major() {
    let (group, e) = (4usize, 3usize);
    // 4 tokens of 3 channels, each value encoding (slot, channel).
    let tokens: Vec<f32> = (0..group * e)
        .map(|i| (i / e) as f32 * 10.0 + (i % e) as f32)
        .collect();

    let mut win = vec![0.0f32; group * e];
    merge_window(&tokens, 0, group, e, &mut win);

    for slot in 0..group {
        for c in 0..e {
            assert_eq!(
                win[c * group + slot],
                slot as f32 * 10.0 + c as f32,
                "win[c={c} * group + slot={slot}] must hold token {slot} channel {c}"
            );
        }
    }
    // The token-major misreading would put slot 1 channel 2 at index 5;
    // channel-major puts it at 2*4 + 1 = 9.
    assert_eq!(win[9], 12.0, "slot 1, channel 2");
    assert_ne!(win[5], 12.0, "index 5 is the token-major misreading");

    // A second group reads the next four tokens.
    let two: Vec<f32> = (0..2 * group * e).map(|i| i as f32).collect();
    let mut w2 = vec![0.0f32; group * e];
    merge_window(&two, 1, group, e, &mut w2);
    assert_eq!(w2[0], (group * e) as f32, "group 1 starts at token 4");
}

// --- preprocessing ---------------------------------------------------

/// The released geometry: factor 28, token budget 16..8000.
fn sr(w: usize, h: usize) -> (usize, usize) {
    smart_resize(w, h, 14, 2, MIN_IMAGE_TOKENS, MAX_IMAGE_TOKENS).unwrap()
}

#[test]
fn patch_area_is_the_token_footprint() {
    assert_eq!(patch_area(14, 2), 784);
    // min/max pixels, as set_limit_image_tokens computes them.
    assert_eq!(MIN_IMAGE_TOKENS * patch_area(14, 2), 12_544);
    assert_eq!(MAX_IMAGE_TOKENS * patch_area(14, 2), 6_272_000);
}

/// A canvas already inside the budget and aligned is returned unchanged.
/// 448 = 16 x 28, the mmproj's nominal image_size.
#[test]
fn an_aligned_in_budget_canvas_is_unchanged() {
    assert_eq!(sr(448, 448), (448, 448));
    assert_eq!(n_image_tokens(448, 448, 14, 2), 256);
}

/// A tiny image is scaled UP to the minimum token count, landing exactly on
/// it: sqrt(12544) = 112 = 4 x 28, giving 16 tokens.
#[test]
fn a_tiny_image_is_scaled_up_to_the_minimum() {
    assert_eq!(sr(1, 1), (112, 112));
    assert_eq!(n_image_tokens(112, 112, 14, 2), MIN_IMAGE_TOKENS);
}

#[test]
fn edges_are_always_aligned_to_the_factor() {
    for (w, h) in [(1, 1), (27, 29), (100, 100), (450, 300), (4000, 3000), (10000, 100)] {
        let (aw, ah) = sr(w, h);
        assert_eq!(aw % 28, 0, "{w}x{h} -> width {aw} is not aligned");
        assert_eq!(ah % 28, 0, "{w}x{h} -> height {ah} is not aligned");
        assert!(aw > 0 && ah > 0);
    }
}

/// Every canvas must sit inside the token budget, including the ones that
/// need the binary search.
#[test]
fn every_canvas_is_inside_the_token_budget() {
    for (w, h) in [
        (1, 1),
        (27, 29),
        (448, 448),
        (4000, 3000),
        (8000, 8000),
        (20000, 17),
        (17, 20000),
    ] {
        let (aw, ah) = sr(w, h);
        let tok = n_image_tokens(aw, ah, 14, 2);
        assert!(
            tok <= MAX_IMAGE_TOKENS,
            "{w}x{h} -> {aw}x{ah} = {tok} tokens, over the budget"
        );
        assert!(tok >= 1, "{w}x{h} -> {aw}x{ah} produced no tokens");
    }
}

/// A very large image trips the upper clamp and its binary search, and the
/// result should be close to the budget rather than collapsing to the
/// one-block fallback.
#[test]
fn a_large_image_lands_near_the_budget() {
    let (aw, ah) = sr(4000, 3000);
    let tok = n_image_tokens(aw, ah, 14, 2);
    assert!(tok <= MAX_IMAGE_TOKENS);
    assert!(
        tok > MAX_IMAGE_TOKENS / 2,
        "the search should get near the budget, got {tok} tokens ({aw}x{ah})"
    );
    // Aspect ratio roughly preserved.
    let want = 4000.0f64 / 3000.0;
    let got = aw as f64 / ah as f64;
    assert!((got - want).abs() < 0.05, "aspect {got} vs {want}");
}

#[test]
fn smart_resize_rejects_bad_parameters_and_handles_empty() {
    assert_eq!(smart_resize(0, 10, 14, 2, 16, 8000).unwrap(), (0, 0));
    assert_eq!(smart_resize(10, 0, 14, 2, 16, 8000).unwrap(), (0, 0));
    assert!(smart_resize(10, 10, 0, 2, 16, 8000).is_err());
    assert!(smart_resize(10, 10, 14, 2, 0, 8000).is_err());
    assert!(smart_resize(10, 10, 14, 2, 100, 10).is_err(), "inverted range");
}

/// End to end from encoded bytes: preprocess, M-RoPE, tower, projector.
#[test]
fn encode_image_bytes_runs_end_to_end() {
    let sh = shape();
    let o = weights(&sh);
    let w = model(&sh, &o);

    // A small PNG built in memory; the fixture's patch is 2 and n_merge 2,
    // so the factor is 4 and a 16x12 canvas is already aligned.
    let mut raw = image::RgbImage::new(16, 12);
    for (x, y, px) in raw.enumerate_pixels_mut() {
        *px = image::Rgb([(x * 13 % 256) as u8, (y * 29 % 256) as u8, ((x + y) % 256) as u8]);
    }
    let mut bytes = Vec::new();
    image::DynamicImage::ImageRgb8(raw)
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("encode png");

    let (feats, n_tok) = encode_image_bytes(&sh, &w, &bytes, 10000.0).unwrap();
    assert!(n_tok > 0, "the image must produce tokens");
    assert_eq!(feats.len(), n_tok * sh.proj_dim);
    assert!(feats.iter().all(|x| x.is_finite()), "features must be finite");
    assert!(feats.iter().any(|&x| x != 0.0));

    assert!(encode_image_bytes(&sh, &w, b"not an image", 10000.0).is_err());
}

/// Normalisation must centre the data: a mid-grey image maps near zero for
/// each channel given CLIP's constants.
#[test]
fn preprocess_normalises_per_channel() {
    let sh = shape();
    let mut raw = image::RgbImage::new(8, 8);
    for px in raw.pixels_mut() {
        *px = image::Rgb([123, 117, 104]); // ~ 255 * IMAGE_MEAN
    }
    let img = image::DynamicImage::ImageRgb8(raw);
    let (buf, w, h) = preprocess(&img, &sh, IMAGE_MEAN, IMAGE_STD).unwrap();
    assert_eq!(buf.len(), 3 * w * h);
    for c in 0..3 {
        let plane = &buf[c * w * h..(c + 1) * w * h];
        let mean = plane.iter().sum::<f32>() / plane.len() as f32;
        assert!(mean.abs() < 0.05, "channel {c} mean {mean} should be near 0");
    }
    // A zero std is rejected rather than producing infinities.
    assert!(preprocess(&img, &sh, IMAGE_MEAN, [1.0, 0.0, 1.0]).is_err());
}

// --- M-RoPE ----------------------------------------------------------

fn rope_for(pos: Vec<(i32, i32)>) -> MRope {
    MRope::new(8, 10000.0, pos).unwrap()
}

#[test]
fn a_zero_position_is_the_identity() {
    let r = rope_for(vec![(0, 0)]);
    let mut buf = fill(2 * 8, 31);
    let before = buf.clone();
    r.rotate(&mut buf, 0, 2, 8);
    for (i, (a, b)) in buf.iter().zip(before.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "element {i} moved at position 0");
    }
}

/// A rotation is orthogonal, so each (i, i + n_dims) pair keeps its norm.
#[test]
fn rotation_preserves_each_pairs_norm() {
    let r = rope_for(vec![(7, 3)]);
    let mut buf = fill(8, 33);
    let before = buf.clone();
    r.rotate(&mut buf, 0, 1, 8);
    let n = 4; // d_head / 2
    for i in 0..n {
        let a = before[i] * before[i] + before[i + n] * before[i + n];
        let b = buf[i] * buf[i] + buf[i + n] * buf[i + n];
        assert!((a - b).abs() < 1e-5, "pair {i} norm changed: {a} vs {b}");
    }
}

/// Pair 0's scale exponent is 0, so it rotates by exactly the row.
#[test]
fn pair_zero_rotates_by_exactly_the_row() {
    let r = rope_for(vec![(1, 0)]);
    let mut buf = vec![0.0f32; 8];
    buf[0] = 1.0; // pair 0 is (buf[0], buf[4])
    r.rotate(&mut buf, 0, 1, 8);
    assert!((buf[0] - 1.0f32.cos()).abs() < 1e-6, "got {}", buf[0]);
    assert!((buf[4] - 1.0f32.sin()).abs() < 1e-6, "got {}", buf[4]);
}

/// The row feeds the first `d_head/4` pairs and the column the rest. Changing
/// only the column must leave the row section untouched, and vice versa.
#[test]
fn the_row_and_column_sections_are_independent() {
    let sec0 = 2; // d_head / 4 for d_head = 8
    let n = 4; // n_dims
    let base = fill(8, 41);

    let r = rope_for(vec![(5, 0), (5, 9), (2, 0)]);
    let mut same_row_diff_col = base.clone();
    let mut ref_tok = base.clone();
    let mut diff_row_same_col = base.clone();
    r.rotate(&mut ref_tok, 0, 1, 8); // (5, 0)
    r.rotate(&mut same_row_diff_col, 1, 1, 8); // (5, 9)
    r.rotate(&mut diff_row_same_col, 2, 1, 8); // (2, 0)

    // Row section: pairs 0..sec0 -- unchanged when only the column moves.
    for i in 0..sec0 {
        assert!(
            (ref_tok[i] - same_row_diff_col[i]).abs() < 1e-6
                && (ref_tok[i + n] - same_row_diff_col[i + n]).abs() < 1e-6,
            "pair {i} is in the ROW section and must ignore the column"
        );
    }
    // Column section: pairs sec0..n -- unchanged when only the row moves.
    for i in sec0..n {
        assert!(
            (ref_tok[i] - diff_row_same_col[i]).abs() < 1e-6
                && (ref_tok[i + n] - diff_row_same_col[i + n]).abs() < 1e-6,
            "pair {i} is in the COLUMN section and must ignore the row"
        );
    }
    // And each section does respond to its own component.
    assert!(
        (0..sec0).any(|i| (ref_tok[i] - diff_row_same_col[i]).abs() > 1e-4),
        "the row section must respond to the row"
    );
    assert!(
        (sec0..n).any(|i| (ref_tok[i] - same_row_diff_col[i]).abs() > 1e-4),
        "the column section must respond to the column"
    );
}

/// The defining RoPE property: after rotation, a query/key pair's dot product
/// depends only on the **difference** of their positions.
#[test]
fn rotated_dot_depends_only_on_the_position_difference() {
    let n = 4;
    let q0 = fill(8, 51);
    let k0 = fill(8, 53);
    // Two token pairs with the same row delta (2) and the same column (0).
    let r = rope_for(vec![(5, 0), (3, 0), (7, 0), (5, 0)]);

    let dots = |qi: usize, ki: usize| -> Vec<f32> {
        let mut q = q0.clone();
        let mut k = k0.clone();
        r.rotate(&mut q, qi, 1, 8);
        r.rotate(&mut k, ki, 1, 8);
        (0..n).map(|i| q[i] * k[i] + q[i + n] * k[i + n]).collect()
    };
    let a = dots(0, 1); // rows 5 vs 3
    let b = dots(2, 3); // rows 7 vs 5
    for i in 0..n {
        assert!(
            (a[i] - b[i]).abs() < 1e-5,
            "pair {i}: {} vs {} -- the dot must depend only on the delta",
            a[i],
            b[i]
        );
    }
}

/// `vision_positions` and `reordered_index` are independent derivations of
/// the same token order -- from clip's position-fill loop and from GLM-4V's
/// permute dance respectively. They must agree.
#[test]
fn positions_agree_with_the_patch_reorder() {
    let (nx, ny, m) = (6usize, 4usize, 2usize);
    let pos = vision_positions(nx, ny, m);
    assert_eq!(pos.len(), nx * ny);
    for py in 0..ny {
        for px in 0..nx {
            let t = reordered_index(px, py, nx, m);
            assert_eq!(
                pos[t],
                (py as i32, px as i32),
                "token {t} should carry patch ({px},{py})"
            );
        }
    }
}

#[test]
fn mrope_changes_the_tower_output() {
    let sh = shape();
    let o = weights(&sh);
    let w = model(&sh, &o);
    let (iw, ih) = (8usize, 8usize);
    let img = fill(3 * iw * ih, 5);
    let (nx, ny) = (iw / sh.patch, ih / sh.patch);

    let blind = encode_image(&sh, &w, &NoRope, &img, iw, ih).unwrap();
    let rope = MRope::for_grid(sh.d_head(), 10000.0, nx, ny, sh.n_merge).unwrap();
    let posd = encode_image(&sh, &w, &rope, &img, iw, ih).unwrap();

    assert!(posd.iter().all(|x| x.is_finite()));
    let diff: f32 = blind.iter().zip(&posd).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "M-RoPE must change the features, diff {diff}");
}

#[test]
fn mrope_rejects_a_bad_head_size() {
    assert!(MRope::new(0, 10000.0, vec![]).is_err());
    assert!(MRope::new(6, 10000.0, vec![]).is_err(), "not a multiple of 4");
    assert!(MRope::new(8, 0.0, vec![]).is_err(), "freq_base must be positive");
    // An out-of-range token is a no-op, not a panic.
    let r = rope_for(vec![(1, 1)]);
    let mut buf = vec![1.0f32; 8];
    r.rotate(&mut buf, 99, 1, 8);
    assert_eq!(buf, vec![1.0f32; 8]);
}

/// The whole point of this module's own SwiGLU: the vision tower clamps the
/// gate BEFORE the SiLU, the text FFN clamps the activation AFTER it. With a
/// gate above the limit the two disagree.
#[test]
fn clamp_order_differs_from_the_text_model() {
    let limit = 2.0f32;
    let gate = vec![5.0f32];
    let up = vec![1.0f32];

    let mut vision = vec![0.0f32; 1];
    swiglu_clamped_pre(&gate, &up, limit, &mut vision).unwrap();
    // clamp(5) = 2, silu(2) = 1.7616
    assert!((vision[0] - silu(2.0)).abs() < 1e-6, "got {}", vision[0]);

    let mut text = vec![0.0f32; 1];
    crate::glm5next::forward::swiglu_clamped(&gate, &up, limit, &mut text).unwrap();
    // silu(5) = 4.9665 -> clamped to 2.0
    assert!((text[0] - 2.0).abs() < 1e-6, "got {}", text[0]);

    assert!(
        (vision[0] - text[0]).abs() > 0.2,
        "the two clamp orders must differ: {} vs {}",
        vision[0],
        text[0]
    );
}

#[test]
fn swiglu_clamps_up_symmetrically_and_gate_one_sided() {
    let limit = 3.0f32;
    // A very negative gate is left alone (one-sided); up is clamped both ways.
    let mut out = vec![0.0f32; 2];
    swiglu_clamped_pre(&[-50.0, 1.0], &[100.0, -100.0], limit, &mut out).unwrap();
    assert!(out[0].abs() < 1e-4, "silu(-50) ~ 0, got {}", out[0]);
    assert!((out[1] - silu(1.0) * -3.0).abs() < 1e-5, "got {}", out[1]);
    assert!(swiglu_clamped_pre(&[1.0], &[1.0], 0.0, &mut vec![0.0; 1]).is_err());
}

/// Each `n_merge x n_merge` spatial block must land on `n_merge^2`
/// consecutive sequence slots, or the projector's merger reads the wrong
/// patches.
#[test]
fn the_reorder_makes_each_block_consecutive() {
    let (nx, ny, m) = (4usize, 4usize, 2usize);
    let mut seen = vec![usize::MAX; nx * ny];
    for py in 0..ny {
        for px in 0..nx {
            let t = reordered_index(px, py, nx, m);
            assert!(t < nx * ny, "index {t} out of range");
            assert_eq!(seen[t], usize::MAX, "slot {t} claimed twice");
            seen[t] = py * nx + px;
        }
    }
    assert!(seen.iter().all(|&s| s != usize::MAX), "every slot filled");

    // Group g holds the 2x2 block at (bx, by) = (g % 2, g / 2).
    for g in 0..(nx * ny) / (m * m) {
        let blocks_x = nx / m;
        let (bx, by) = (g % blocks_x, g / blocks_x);
        let mut members: Vec<(usize, usize)> = (0..m * m)
            .map(|i| {
                let src = seen[g * m * m + i];
                (src % nx, src / nx)
            })
            .collect();
        members.sort();
        let mut want: Vec<(usize, usize)> = Vec::new();
        for dy in 0..m {
            for dx in 0..m {
                want.push((bx * m + dx, by * m + dy));
            }
        }
        want.sort();
        assert_eq!(members, want, "group {g} is not the 2x2 block at ({bx},{by})");
    }
}

/// The two patch convolutions are summed, so zeroing one halves nothing --
/// but replacing one with the negation of the other must give exactly the
/// patch bias.
#[test]
fn the_two_patch_convolutions_are_summed() {
    let sh = shape();
    let kern = 3 * sh.patch * sh.patch;
    let w0 = fill(sh.n_embd * kern, 1);
    let neg: Vec<f32> = w0.iter().map(|v| -v).collect();
    let bias = vec![0.25f32; sh.n_embd];
    let blocks = Vec::new();
    let zeros = vec![0.0f32; 1];

    let w = VisionW {
        patch_embd_0: &w0,
        patch_embd_1: &neg,
        patch_bias: &bias,
        blocks,
        post_ln: &zeros,
        merger: &zeros,
        merger_b: &zeros,
        fc: &zeros,
        post_norm: &zeros,
        post_norm_b: &zeros,
        gate: &zeros,
        up: &zeros,
        down: &zeros,
    };
    let (iw, ih) = (sh.patch * sh.n_merge, sh.patch * sh.n_merge);
    let img = fill(3 * iw * ih, 7);
    let out = patch_embed(&sh, &w, &img, iw, ih).unwrap();
    assert_eq!(out.len(), (iw / sh.patch) * (ih / sh.patch) * sh.n_embd);
    for (i, v) in out.iter().enumerate() {
        assert!((v - 0.25).abs() < 1e-5, "element {i} = {v}, want the bias");
    }
}

#[test]
fn patch_embed_rejects_a_misaligned_image() {
    let sh = shape();
    let kern = 3 * sh.patch * sh.patch;
    let w0 = fill(sh.n_embd * kern, 1);
    let bias = vec![0.0f32; sh.n_embd];
    let zeros = vec![0.0f32; 1];
    let w = VisionW {
        patch_embd_0: &w0,
        patch_embd_1: &w0,
        patch_bias: &bias,
        blocks: Vec::new(),
        post_ln: &zeros,
        merger: &zeros,
        merger_b: &zeros,
        fc: &zeros,
        post_norm: &zeros,
        post_norm_b: &zeros,
        gate: &zeros,
        up: &zeros,
        down: &zeros,
    };
    // patch * n_merge = 4, so 6 is misaligned.
    let img = vec![0.0f32; 3 * 6 * 4];
    assert!(patch_embed(&sh, &w, &img, 6, 4).is_err());
}

#[test]
fn gelu_erf_matches_known_values() {
    assert!((gelu_erf(0.0)).abs() < 1e-7);
    // 0.5 * 1 * (1 + erf(1/sqrt(2))) = 0.8413447
    assert!((gelu_erf(1.0) - 0.8413447).abs() < 1e-5, "{}", gelu_erf(1.0));
    assert!((gelu_erf(-1.0) - -0.1586553).abs() < 1e-5, "{}", gelu_erf(-1.0));
    assert!((gelu_erf(3.0) - 2.9959502).abs() < 1e-4, "{}", gelu_erf(3.0));
    // erf is odd and bounded.
    assert!((erf(0.0)).abs() < 1e-7);
    assert!((erf(2.0) - 0.9953223).abs() < 2e-6);
    assert!((erf(-2.0) + 0.9953223).abs() < 2e-6);
}

// --- a whole tiny tower -------------------------------------------------

struct Owned {
    bufs: std::collections::BTreeMap<String, Vec<f32>>,
}

fn weights(sh: &VisionShape) -> Owned {
    let e = sh.n_embd;
    let kern = 3 * sh.patch * sh.patch;
    let group = sh.n_merge * sh.n_merge;
    let mut b: std::collections::BTreeMap<String, Vec<f32>> = Default::default();
    b.insert("pe0".into(), fill(e * kern, 11));
    b.insert("pe1".into(), fill(e * kern, 13));
    b.insert("pbias".into(), vec![0.01f32; e]);
    b.insert("post_ln".into(), vec![1.0f32; e]);
    for il in 0..sh.n_layer {
        let s = 17 + il * 7;
        b.insert(format!("ln1.{il}"), vec![1.0f32; e]);
        b.insert(format!("ln2.{il}"), vec![1.0f32; e]);
        b.insert(format!("qkv.{il}"), fill(3 * e * e, s));
        b.insert(format!("qkv_b.{il}"), vec![0.0f32; 3 * e]);
        b.insert(format!("qn.{il}"), vec![1.0f32; sh.d_head()]);
        b.insert(format!("kn.{il}"), vec![1.0f32; sh.d_head()]);
        b.insert(format!("out.{il}"), fill(e * e, s + 1));
        b.insert(format!("out_b.{il}"), vec![0.0f32; e]);
        b.insert(format!("fg.{il}"), fill(sh.n_ff * e, s + 2));
        b.insert(format!("fg_b.{il}"), vec![0.0f32; sh.n_ff]);
        b.insert(format!("fu.{il}"), fill(sh.n_ff * e, s + 3));
        b.insert(format!("fu_b.{il}"), vec![0.0f32; sh.n_ff]);
        b.insert(format!("fd.{il}"), fill(e * sh.n_ff, s + 4));
        b.insert(format!("fd_b.{il}"), vec![0.0f32; e]);
    }
    b.insert("merger".into(), fill(sh.proj_dim * group * e, 91));
    b.insert("merger_b".into(), vec![0.0f32; sh.proj_dim]);
    b.insert("fc".into(), fill(sh.proj_dim * sh.proj_dim, 93));
    b.insert("pn".into(), vec![1.0f32; sh.proj_dim]);
    b.insert("pn_b".into(), vec![0.0f32; sh.proj_dim]);
    b.insert("pg".into(), fill(sh.proj_ff * sh.proj_dim, 95));
    b.insert("pu".into(), fill(sh.proj_ff * sh.proj_dim, 97));
    b.insert("pd".into(), fill(sh.proj_dim * sh.proj_ff, 99));
    Owned { bufs: b }
}

fn model<'a>(sh: &VisionShape, o: &'a Owned) -> VisionW<'a> {
    let g = |k: &str| -> &'a [f32] { o.bufs[k].as_slice() };
    let blocks = (0..sh.n_layer)
        .map(|il| VitBlockW {
            ln1: g(&format!("ln1.{il}")),
            ln2: g(&format!("ln2.{il}")),
            qkv: g(&format!("qkv.{il}")),
            qkv_b: g(&format!("qkv_b.{il}")),
            q_norm: g(&format!("qn.{il}")),
            k_norm: g(&format!("kn.{il}")),
            out: g(&format!("out.{il}")),
            out_b: g(&format!("out_b.{il}")),
            ffn_gate: g(&format!("fg.{il}")),
            ffn_gate_b: g(&format!("fg_b.{il}")),
            ffn_up: g(&format!("fu.{il}")),
            ffn_up_b: g(&format!("fu_b.{il}")),
            ffn_down: g(&format!("fd.{il}")),
            ffn_down_b: g(&format!("fd_b.{il}")),
        })
        .collect();
    VisionW {
        patch_embd_0: g("pe0"),
        patch_embd_1: g("pe1"),
        patch_bias: g("pbias"),
        blocks,
        post_ln: g("post_ln"),
        merger: g("merger"),
        merger_b: g("merger_b"),
        fc: g("fc"),
        post_norm: g("pn"),
        post_norm_b: g("pn_b"),
        gate: g("pg"),
        up: g("pu"),
        down: g("pd"),
    }
}

#[test]
fn the_tower_encodes_an_image_to_merged_tokens() {
    let sh = shape();
    let o = weights(&sh);
    let w = model(&sh, &o);
    // 4x4 patches -> 16 tokens -> 4 merged tokens.
    let (iw, ih) = (8usize, 8usize);
    let img = fill(3 * iw * ih, 5);

    let out = encode_image(&sh, &w, &NoRope, &img, iw, ih).unwrap();
    let nx = iw / sh.patch;
    let ny = ih / sh.patch;
    assert_eq!(sh.n_out_tokens(nx, ny), 4);
    assert_eq!(out.len(), 4 * sh.proj_dim);
    assert!(out.iter().all(|x| x.is_finite()), "features must be finite");
    assert!(out.iter().any(|&x| x != 0.0), "features must not be all zero");

    // Different images must give different features.
    let img2 = fill(3 * iw * ih, 6);
    let out2 = encode_image(&sh, &w, &NoRope, &img2, iw, ih).unwrap();
    let diff: f32 = out.iter().zip(&out2).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-6, "the tower must depend on its input, diff {diff}");
}

#[test]
fn the_tower_is_deterministic() {
    let sh = shape();
    let o = weights(&sh);
    let w = model(&sh, &o);
    let img = fill(3 * 8 * 8, 5);
    let a = encode_image(&sh, &w, &NoRope, &img, 8, 8).unwrap();
    let b = encode_image(&sh, &w, &NoRope, &img, 8, 8).unwrap();
    assert_eq!(a, b);
}

/// A block is a residual path: zero output projections must leave the hidden
/// state untouched.
#[test]
fn a_block_with_zero_output_projections_is_the_identity() {
    let sh = shape();
    let e = sh.n_embd;
    let zeros_e2 = vec![0.0f32; e * e];
    let zeros_ff = vec![0.0f32; e * sh.n_ff];
    let ones = vec![1.0f32; e];
    let dh1 = vec![1.0f32; sh.d_head()];
    let qkv = fill(3 * e * e, 3);
    let qkv_b = vec![0.0f32; 3 * e];
    let ffw = fill(sh.n_ff * e, 4);
    let ffb = vec![0.0f32; sh.n_ff];
    let zb = vec![0.0f32; e];

    let b = VitBlockW {
        ln1: &ones,
        ln2: &ones,
        qkv: &qkv,
        qkv_b: &qkv_b,
        q_norm: &dh1,
        k_norm: &dh1,
        out: &zeros_e2,
        out_b: &zb,
        ffn_gate: &ffw,
        ffn_gate_b: &ffb,
        ffn_up: &ffw,
        ffn_up_b: &ffb,
        ffn_down: &zeros_ff,
        ffn_down_b: &zb,
    };
    let mut x = fill(4 * e, 9);
    let before = x.clone();
    vit_block(&sh, &b, &NoRope, &mut x, 4).unwrap();
    for (i, (a, c)) in x.iter().zip(before.iter()).enumerate() {
        assert!((a - c).abs() < 1e-6, "element {i} moved: {a} vs {c}");
    }
}

#[test]
fn the_projector_merges_groups_of_four() {
    let sh = shape();
    let o = weights(&sh);
    let w = model(&sh, &o);
    let tokens = fill(8 * sh.n_embd, 21);
    let out = project(&sh, &w, &tokens).unwrap();
    assert_eq!(out.len(), 2 * sh.proj_dim, "8 tokens -> 2 merged");
    // A token count that is not a multiple of n_merge^2 is rejected.
    let ragged = fill(6 * sh.n_embd, 21);
    assert!(project(&sh, &w, &ragged).is_err());
}

#[test]
fn encode_rejects_a_wrong_block_count() {
    let mut sh = shape();
    let o = weights(&sh);
    let w = model(&sh, &o);
    sh.n_layer = 3; // weights only have 2
    let img = fill(3 * 8 * 8, 5);
    assert!(encode_image(&sh, &w, &NoRope, &img, 8, 8).is_err());
}
