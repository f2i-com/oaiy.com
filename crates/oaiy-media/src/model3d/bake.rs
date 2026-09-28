//! Baked PBR textures (the bake in TRELLIS.2's `to_glb`): the simplified mesh is cut
//! into charts, each laid flat and packed into one square atlas; every texel a
//! triangle covers finds its point on the decoded surface and samples the texture
//! voxels there; then the space between charts is filled, so bilinear filtering and
//! mipmaps never reach the background.
//!
//! Only the faces that show get charts. The remeshed shell's inside sheet (about
//! half its area) and faces too small to be worth a chart of their own are coloured
//! per vertex instead (COLOR_0, which glTF multiplies with the texture) and point at
//! a white patch of the atlas: were one of them to show after all, it has its colour.
//!
//! A chart grows across edges while its faces stay within 60° of its mean normal,
//! and is projected onto the plane across that normal. A face that would overlap
//! another in its chart's projection (where layers stack, as in braids), or would
//! lie nearly edge-on in it, starts a chart of its own. This stands in for CuMesh's
//! chart clustering and xatlas, which takes minutes on a 200k-face mesh on the CPU;
//! the charts pack by rows.
use super::glb::Textures;
use super::mesh::{Mesh, Voxels};
use super::remesh::{par_map, Surface};
use candle_core::Result;
use std::collections::{HashMap, VecDeque};

/// Texels between charts (half on each side of one).
const PADDING: u32 = 2;
/// How far a face may turn from its chart's mean normal: the cosine of 60°.
const CONE: f32 = 0.5;
/// How edge-on a face may lie in its chart's projection (the cosine of its tilt).
const TILT: f32 = 0.25;
/// A face alone in its chart covering fewer texels than this is coloured per vertex.
const SCRAP: f32 = 6.;
/// The white patch's side, in texels.
const WHITE: u32 = 4;
/// How far (in voxels of the surface's grid) a texel's point looks for the decoded surface.
const REACH: f32 = 3.;
/// Directions the model is looked at from to find the faces that show, and their
/// depth buffers' side.
const VIEWS: usize = 96;
const VIEW_SIZE: usize = 1024;

/// The mesh cut along its charts' seams, with its textures.
pub struct Baked {
    pub mesh: Mesh,
    pub textures: Textures,
    pub charts: usize,
    /// Faces coloured per vertex rather than from a chart.
    pub painted: usize,
    /// The share of the atlas the charts' triangles cover.
    pub coverage: f32,
}

/// Part of the mesh laid flat.
struct Chart {
    /// The mesh's vertices it holds, and their places on its plane (from 0, in mesh units).
    verts: Vec<u32>,
    flat: Vec<[f32; 2]>,
    /// Its faces, and their corners as indices into `verts`.
    faces: Vec<u32>,
    tris: Vec<[u32; 3]>,
    size: [f32; 2],
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    if l > 0. {
        v.map(|x| x / l)
    } else {
        [0., 0., 1.]
    }
}
fn cross2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}
fn flat_area(p: [[f32; 2]; 3]) -> f32 {
    0.5 * cross2([p[1][0] - p[0][0], p[1][1] - p[0][1]], [p[2][0] - p[0][0], p[2][1] - p[0][1]]).abs()
}

/// Each face's unit normal and area.
fn faces(mesh: &Mesh) -> (Vec<[f32; 3]>, Vec<f32>) {
    mesh.triangles
        .iter()
        .map(|t| {
            let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
            let n = cross(sub(b, a), sub(c, a));
            (normalize(n), 0.5 * dot(n, n).sqrt())
        })
        .unzip()
}

/// Each face's neighbours across its three edges (`u32::MAX` across an open or non-manifold edge).
fn neighbours(mesh: &Mesh) -> Vec<[u32; 3]> {
    let mut edges: HashMap<u64, (u32, u32, u8)> = HashMap::with_capacity(mesh.triangles.len() * 2);
    let edge = |a: u32, b: u32| ((a.min(b) as u64) << 32) | a.max(b) as u64;
    for (f, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let e = edges.entry(edge(t[k], t[(k + 1) % 3])).or_insert((f as u32, u32::MAX, 0));
            if e.2 == 1 {
                e.1 = f as u32;
            }
            e.2 = e.2.saturating_add(1);
        }
    }
    mesh.triangles
        .iter()
        .enumerate()
        .map(|(f, t)| {
            let mut n = [u32::MAX; 3];
            for k in 0..3 {
                let &(a, b, count) = &edges[&edge(t[k], t[(k + 1) % 3])];
                if count == 2 {
                    n[k] = if a == f as u32 { b } else { a };
                }
            }
            n
        })
        .collect()
}

/// Calls `f(x, y, weights)` for each texel of a `w` × `h` grid whose centre is in the
/// triangle `p` (texel units), with its barycentric weights; `margin` widens
/// (negative) or narrows (positive) the triangle, in weight.
fn raster(p: [[f32; 2]; 3], w: usize, h: usize, margin: f32, mut f: impl FnMut(usize, usize, [f32; 3])) {
    let [a, b, c] = p;
    let area = cross2([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
    if area.abs() < 1e-12 {
        return;
    }
    let lo = |k: usize| (a[k].min(b[k]).min(c[k]) - 0.5).floor().max(0.) as usize;
    let hi = |k: usize, n: usize| ((a[k].max(b[k]).max(c[k]) - 0.5).ceil().max(0.) as usize).min(n.saturating_sub(1));
    for y in lo(1)..=hi(1, h) {
        for x in lo(0)..=hi(0, w) {
            let q = [x as f32 + 0.5, y as f32 + 0.5];
            let wa = cross2([c[0] - b[0], c[1] - b[1]], [q[0] - b[0], q[1] - b[1]]) / area;
            let wb = cross2([a[0] - c[0], a[1] - c[1]], [q[0] - c[0], q[1] - c[1]]) / area;
            let wc = 1. - wa - wb;
            if wa >= margin && wb >= margin && wc >= margin {
                f(x, y, [wa, wb, wc]);
            }
        }
    }
}

/// Which faces show from outside: the mesh is drawn from `VIEWS` directions spread
/// over the sphere (orthographic, both sides of every face), and a face shows when
/// it is the nearest at some pixel of some view.
fn visible(mesh: &Mesh) -> Vec<bool> {
    let nf = mesh.triangles.len();
    let golden = std::f32::consts::PI * (3. - 5f32.sqrt());
    let views: Vec<[f32; 3]> = (0..VIEWS)
        .map(|i| {
            let y = 1. - 2. * (i as f32 + 0.5) / VIEWS as f32;
            let r = (1. - y * y).sqrt();
            let (s, c) = (golden * i as f32).sin_cos();
            [c * r, y, s * r]
        })
        .collect();
    // The model fits the unit cube, whose shadow is within √3/2 of its middle.
    let reach = 0.87f32;
    let scale = VIEW_SIZE as f32 / (2. * reach);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).clamp(1, 16);
    let seen: Vec<Vec<bool>> = std::thread::scope(|s| {
        let workers: Vec<_> = views
            .chunks(VIEWS.div_ceil(threads))
            .map(|chunk| {
                s.spawn(move || {
                    let mut shows = vec![false; nf];
                    let mut depth = vec![f32::NEG_INFINITY; VIEW_SIZE * VIEW_SIZE];
                    let mut nearest = vec![u32::MAX; VIEW_SIZE * VIEW_SIZE];
                    for &d in chunk {
                        let helper = if d[0].abs() < 0.9 { [1., 0., 0.] } else { [0., 1., 0.] };
                        let u = normalize(cross(helper, d));
                        let v = cross(d, u);
                        let screen: Vec<[f32; 3]> = mesh.positions.iter().map(|&p| [(dot(p, u) + reach) * scale, (reach - dot(p, v)) * scale, dot(p, d)]).collect();
                        depth.fill(f32::NEG_INFINITY);
                        nearest.fill(u32::MAX);
                        for (f, t) in mesh.triangles.iter().enumerate() {
                            let p = t.map(|i| screen[i as usize]);
                            raster(p.map(|q| [q[0], q[1]]), VIEW_SIZE, VIEW_SIZE, 0., |x, y, w| {
                                let z = w[0] * p[0][2] + w[1] * p[1][2] + w[2] * p[2][2];
                                let k = y * VIEW_SIZE + x;
                                if z > depth[k] {
                                    depth[k] = z;
                                    nearest[k] = f as u32;
                                }
                            });
                        }
                        for &f in &nearest {
                            if f != u32::MAX {
                                shows[f as usize] = true;
                            }
                        }
                    }
                    shows
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().expect("a view panicked")).collect()
    });
    (0..nf).map(|f| seen.iter().any(|s| s[f])).collect()
}

/// Cuts the faces that show into charts, each flat at about `density` texels per
/// unit; the faces left over (hidden, or scraps) are listed to be coloured per vertex.
fn charts(mesh: &Mesh, density: f32, shows: &[bool]) -> (Vec<Chart>, Vec<u32>) {
    let (normals, areas) = faces(mesh);
    let adjacent = neighbours(mesh);
    let nf = mesh.triangles.len();
    // A chart's index, or MAX while a face has none.
    let mut owner = vec![u32::MAX; nf];
    let mut painted: Vec<u32> = (0..nf as u32).filter(|&f| !shows[f as usize]).collect();
    for &f in &painted {
        owner[f as usize] = u32::MAX - 1;
    }
    let mut order: Vec<u32> = (0..nf as u32).rev().collect();
    let mut queue = VecDeque::new();
    let mut slot = vec![u32::MAX; mesh.positions.len()];
    let mut charts = Vec::new();
    while let Some(seed) = order.pop() {
        if owner[seed as usize] != u32::MAX {
            continue;
        }
        let id = charts.len() as u32;
        let area = |f: u32| areas[f as usize].max(1e-12);
        let mut sum = normals[seed as usize].map(|v| v * area(seed));
        let mut grown = vec![seed];
        owner[seed as usize] = id;
        queue.push_back(seed);
        while let Some(f) = queue.pop_front() {
            let mean = normalize(sum);
            for &g in &adjacent[f as usize] {
                if g == u32::MAX || owner[g as usize] != u32::MAX || dot(normals[g as usize], mean) < CONE {
                    continue;
                }
                owner[g as usize] = id;
                for k in 0..3 {
                    sum[k] += normals[g as usize][k] * area(g);
                }
                grown.push(g);
                queue.push_back(g);
            }
        }
        // The plane across the mean normal.
        let n = normalize(sum);
        let helper = if n[0].abs() < 0.9 { [1., 0., 0.] } else { [0., 1., 0.] };
        let u = normalize(cross(helper, n));
        let v = cross(n, u);
        let mut verts = Vec::new();
        let mut flat = Vec::new();
        for &f in &grown {
            for i in mesh.triangles[f as usize] {
                if slot[i as usize] == u32::MAX {
                    slot[i as usize] = verts.len() as u32;
                    verts.push(i);
                    let p = mesh.positions[i as usize];
                    flat.push([dot(p, u), dot(p, v)]);
                }
            }
        }
        let lo = flat.iter().fold([f32::INFINITY; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
        let hi = flat.iter().fold([f32::NEG_INFINITY; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]);
        // Faces overlapping one already laid, or nearly edge-on, leave for charts of their own.
        let (w, h) = (((hi[0] - lo[0]) * density).ceil() as usize + 1, ((hi[1] - lo[1]) * density).ceil() as usize + 1);
        let mut taken = vec![false; w * h];
        let mut faces = Vec::new();
        let mut tris = Vec::new();
        let mut covered = Vec::new();
        for (k, &f) in grown.iter().enumerate() {
            let t = mesh.triangles[f as usize].map(|i| slot[i as usize]);
            let upright = areas[f as usize] <= 1e-12 || dot(normals[f as usize], n).abs() >= TILT;
            covered.clear();
            if k == 0 || upright {
                raster(t.map(|i| [(flat[i as usize][0] - lo[0]) * density, (flat[i as usize][1] - lo[1]) * density]), w, h, 1e-4, |x, y, _| covered.push(y * w + x));
            }
            if k == 0 || (upright && covered.iter().all(|&i| !taken[i])) {
                for &i in &covered {
                    taken[i] = true;
                }
                faces.push(f);
                tris.push(t);
            } else {
                owner[f as usize] = u32::MAX;
                order.push(f);
            }
        }
        for &i in &verts {
            slot[i as usize] = u32::MAX;
        }
        // A face alone, and small, is not worth a chart.
        if faces.len() == 1 && flat_area(tris[0].map(|i| flat[i as usize])) * density * density < SCRAP {
            owner[faces[0] as usize] = u32::MAX - 1;
            painted.push(faces[0]);
            continue;
        }
        // The kept faces' vertices, turned so the chart's long axis runs along u, from 0.
        let mut used = vec![u32::MAX; verts.len()];
        let mut kept_verts = Vec::new();
        let mut kept = Vec::new();
        for t in &mut tris {
            for i in t.iter_mut() {
                if used[*i as usize] == u32::MAX {
                    used[*i as usize] = kept_verts.len() as u32;
                    kept_verts.push(verts[*i as usize]);
                    kept.push(flat[*i as usize]);
                }
                *i = used[*i as usize];
            }
        }
        let count = kept.len() as f32;
        let mean = kept.iter().fold([0f32; 2], |m, p| [m[0] + p[0] / count, m[1] + p[1] / count]);
        let (mut xx, mut xy, mut yy) = (0f32, 0f32, 0f32);
        for p in &kept {
            let d = [p[0] - mean[0], p[1] - mean[1]];
            xx += d[0] * d[0];
            xy += d[0] * d[1];
            yy += d[1] * d[1];
        }
        let (s, c) = (0.5 * (2. * xy).atan2(xx - yy)).sin_cos();
        for p in &mut kept {
            *p = [p[0] * c + p[1] * s, -p[0] * s + p[1] * c];
        }
        let lo = kept.iter().fold([f32::INFINITY; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
        for p in &mut kept {
            *p = [p[0] - lo[0], p[1] - lo[1]];
        }
        let size = kept.iter().fold([0f32; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]);
        charts.push(Chart { verts: kept_verts, flat: kept, faces, tris, size });
    }
    (charts, painted)
}

/// Places `sizes` (texels, padding included) in rows, in `order`, in a `side`² square.
fn rows(sizes: &[[u32; 2]], order: &[usize], side: u32) -> Option<Vec<[u32; 2]>> {
    let mut at = vec![[0u32; 2]; sizes.len()];
    let (mut x, mut y, mut row) = (0u32, 0u32, 0u32);
    for &i in order {
        let [w, h] = sizes[i];
        if w > side {
            return None;
        }
        if x + w > side {
            y += row;
            x = 0;
            row = 0;
        }
        if y + h > side {
            return None;
        }
        at[i] = [x, y];
        x += w;
        row = row.max(h);
    }
    Some(at)
}

/// Packs the charts, tallest first, and the white patch after them into a `side`²
/// atlas at the largest density that fits: each chart's corner in texels, the
/// white patch's, and the density.
fn pack(charts: &mut [Chart], side: u32) -> Result<(Vec<[u32; 2]>, [u32; 2], f32)> {
    // Charts lie wide, so rows stay low.
    for c in charts.iter_mut() {
        if c.size[1] > c.size[0] {
            for p in &mut c.flat {
                *p = [c.size[1] - p[1], p[0]];
            }
            c.size = [c.size[1], c.size[0]];
        }
    }
    let mut order: Vec<usize> = (0..charts.len()).collect();
    order.sort_by(|&a, &b| charts[b].size[1].total_cmp(&charts[a].size[1]));
    order.push(charts.len());
    let sizes = |d: f32| -> Vec<[u32; 2]> {
        let mut s: Vec<[u32; 2]> = charts.iter().map(|c| c.size.map(|s| (s * d).ceil() as u32 + 1 + PADDING)).collect();
        s.push([WHITE + PADDING; 2]);
        s
    };
    let area: f32 = charts.iter().map(|c| c.size[0] * c.size[1]).sum();
    let widest = charts.iter().map(|c| c.size[0]).fold(0f32, f32::max);
    let (mut lo, mut hi) = (0f32, (side as f32 / widest.max(1e-6)).min(side as f32 / area.max(1e-12).sqrt()));
    let Some(mut best) = rows(&sizes(lo), &order, side) else {
        candle_core::bail!("3d: {} charts do not fit a {side}² texture", charts.len());
    };
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        match rows(&sizes(mid), &order, side) {
            Some(at) => {
                lo = mid;
                best = at;
            }
            None => hi = mid,
        }
    }
    let white = best.pop().expect("the white patch is placed");
    Ok((best, white, lo))
}

/// Fills the texels no triangle covers: rings of their known neighbours' averages
/// first (the texels bilinear filtering reads at a chart's edge), then the rest
/// from ever coarser averages (pull-push), so mipmaps blend charts only into
/// colours near their own.
fn fill(data: &mut [[f32; 6]], known: &mut [bool], side: usize) {
    for _ in 0..PADDING + 1 {
        let before = known.to_vec();
        for y in 0..side {
            for x in 0..side {
                if before[y * side + x] {
                    continue;
                }
                let mut sum = [0f32; 6];
                let mut n = 0f32;
                for (dx, dy) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= side as i64 || ny >= side as i64 || !before[ny as usize * side + nx as usize] {
                        continue;
                    }
                    for (s, v) in sum.iter_mut().zip(data[ny as usize * side + nx as usize]) {
                        *s += v;
                    }
                    n += 1.;
                }
                if n > 0. {
                    data[y * side + x] = sum.map(|s| s / n);
                    known[y * side + x] = true;
                }
            }
        }
    }
    pull_push(data, known, side);
}

fn pull_push(data: &mut [[f32; 6]], known: &[bool], side: usize) {
    if side <= 1 || known.iter().all(|k| *k) {
        return;
    }
    let half = side.div_ceil(2);
    let mut coarse = vec![[0f32; 6]; half * half];
    let mut count = vec![0f32; half * half];
    for y in 0..side {
        for x in 0..side {
            if known[y * side + x] {
                let c = (y / 2) * half + x / 2;
                for (s, v) in coarse[c].iter_mut().zip(data[y * side + x]) {
                    *s += v;
                }
                count[c] += 1.;
            }
        }
    }
    let coarse_known: Vec<bool> = count.iter().map(|&n| n > 0.).collect();
    for (c, n) in coarse.iter_mut().zip(&count) {
        if *n > 0. {
            *c = c.map(|v| v / n);
        }
    }
    pull_push(&mut coarse, &coarse_known, half);
    for y in 0..side {
        for x in 0..side {
            if !known[y * side + x] {
                data[y * side + x] = coarse[(y / 2) * half + x / 2];
            }
        }
    }
}

fn png(bytes: &[u8], side: u32, color: image::ExtendedColorType) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new_with_quality(&mut out, image::codecs::png::CompressionType::Fast, image::codecs::png::FilterType::Adaptive)
        .write_image(bytes, side, side, color)
        .map_err(candle_core::Error::wrap)?;
    Ok(out)
}

/// Unwraps `mesh` and bakes its base colour (with alpha) and metallic-roughness
/// textures, `texture_size` texels a side, from the texture voxels at the nearest
/// points of `surface` (the decoded, unsimplified surface).
pub fn bake(mesh: &Mesh, surface: &Surface, voxels: &Voxels, texture_size: u32) -> Result<Baked> {
    if mesh.triangles.is_empty() {
        candle_core::bail!("3d: nothing to bake (the mesh has no triangles)");
    }
    let side = texture_size as usize;
    let sample = |p: [f32; 3]| voxels.sample(surface.closest(p, REACH).unwrap_or(p));
    let shows = visible(mesh);
    let (_, areas) = faces(mesh);
    let area: f32 = areas.iter().zip(&shows).filter(|(_, s)| **s).map(|(a, _)| a).sum();
    // The projection's density, near what the packing will give (about 40% of the atlas covered).
    let (mut charts, painted) = charts(mesh, texture_size as f32 * (0.4 / area.max(1e-12)).sqrt(), &shows);
    let (corners, white, density) = pack(&mut charts, texture_size)?;

    // The split mesh, and each texel's point on it.
    let normals = mesh.normals();
    let mut positions = Vec::new();
    let mut split_normals = Vec::new();
    let mut uvs = Vec::new();
    let mut triangles = Vec::new();
    let mut owner = vec![u32::MAX; side * side];
    let mut points: Vec<(u32, [f32; 3])> = Vec::new();
    let half = (PADDING / 2) as f32;
    for (c, at) in charts.iter().zip(&corners) {
        let base = positions.len() as u32;
        let texel = |p: [f32; 2]| [at[0] as f32 + half + p[0] * density, at[1] as f32 + half + p[1] * density];
        for (&v, &p) in c.verts.iter().zip(&c.flat) {
            positions.push(mesh.positions[v as usize]);
            split_normals.push(normals[v as usize]);
            let t = texel(p);
            uvs.push([t[0] / side as f32, t[1] / side as f32]);
        }
        for (&f, t) in c.faces.iter().zip(&c.tris) {
            triangles.push(t.map(|i| base + i));
            let corners3 = t.map(|i| mesh.positions[c.verts[i as usize] as usize]);
            let mut any = false;
            raster(t.map(|i| texel(c.flat[i as usize])), side, side, -1e-3, |x, y, w| {
                let i = y * side + x;
                any = true;
                if owner[i] == u32::MAX {
                    owner[i] = f;
                    points.push((i as u32, [0, 1, 2].map(|k| w[0] * corners3[0][k] + w[1] * corners3[1][k] + w[2] * corners3[2][k])));
                }
            });
            // A triangle smaller than a texel still gets the one under its middle.
            if !any {
                let m = t.map(|i| texel(c.flat[i as usize]));
                let (x, y) = (((m[0][0] + m[1][0] + m[2][0]) / 3.) as usize, ((m[0][1] + m[1][1] + m[2][1]) / 3.) as usize);
                let i = y.min(side - 1) * side + x.min(side - 1);
                if owner[i] == u32::MAX {
                    owner[i] = f;
                    points.push((i as u32, [0, 1, 2].map(|k| (corners3[0][k] + corners3[1][k] + corners3[2][k]) / 3.)));
                }
            }
        }
    }
    drop(owner);
    let charted = positions.len();

    // The painted faces share their own copies of their vertices, on the white patch.
    let white_uv = [(white[0] as f32 + half + WHITE as f32 / 2.) / side as f32, (white[1] as f32 + half + WHITE as f32 / 2.) / side as f32];
    let mut slot: HashMap<u32, u32> = HashMap::new();
    let mut painted_verts = Vec::new();
    for &f in &painted {
        triangles.push(mesh.triangles[f as usize].map(|v| {
            *slot.entry(v).or_insert_with(|| {
                painted_verts.push(v);
                (charted + painted_verts.len() - 1) as u32
            })
        }));
    }
    let painted_values = par_map(&painted_verts, |&v| sample(mesh.positions[v as usize]));
    for &v in &painted_verts {
        positions.push(mesh.positions[v as usize]);
        split_normals.push(normals[v as usize]);
        uvs.push(white_uv);
    }
    let colors = (!painted.is_empty()).then(|| {
        let mut c = vec![[1f32; 4]; charted];
        c.extend(painted_values.iter().map(|a| [a[0], a[1], a[2], a[5]]));
        c
    });

    // Each texel's values, from the voxels at its point's nearest place on the decoded surface.
    let values = par_map(&points, |(_, p)| sample(*p));
    let mut data = vec![[0f32; 6]; side * side];
    let mut known = vec![false; side * side];
    for ((i, _), v) in points.iter().zip(&values) {
        data[*i as usize] = *v;
        known[*i as usize] = true;
    }
    // The white patch has the painted faces' mean metallic and roughness.
    let n = painted_values.len().max(1) as f32;
    let (metal, rough) = painted_values.iter().fold((0f32, 0f32), |(m, r), a| (m + a[3] / n, r + a[4] / n));
    let (metal, rough) = if painted_values.is_empty() { (0., 1.) } else { (metal, rough) };
    for y in 0..(WHITE + PADDING) as usize {
        for x in 0..(WHITE + PADDING) as usize {
            let i = (white[1] as usize + y) * side + white[0] as usize + x;
            data[i] = [1., 1., 1., metal, rough, 1.];
            known[i] = true;
        }
    }
    let covered = points.len();
    let clear = values.iter().filter(|v| v[5] < 0.5).count();
    fill(&mut data, &mut known, side);
    let byte = |v: f32| (v * 255.).round().clamp(0., 255.) as u8;
    let base: Vec<u8> = data.iter().flat_map(|v| [byte(v[0]), byte(v[1]), byte(v[2]), byte(v[5])]).collect();
    // glTF's layout: roughness in green, metallic in blue.
    let mr: Vec<u8> = data.iter().flat_map(|v| [0, byte(v[4]), byte(v[3])]).collect();
    drop(data);
    let (base_png, mr_png) = std::thread::scope(|s| {
        let b = s.spawn(|| png(&base, texture_size, image::ExtendedColorType::Rgba8));
        let m = png(&mr, texture_size, image::ExtendedColorType::Rgb8);
        (b.join().expect("PNG encoding panicked"), m)
    });
    Ok(Baked {
        mesh: Mesh { positions, triangles, colors, metal_rough: None },
        textures: Textures {
            uvs,
            normals: split_normals,
            base_color_png: base_png?,
            metallic_roughness_png: mr_png?,
            // As TRELLIS.2's export, opaque, unless much of the surface is see-through.
            transparent: clear * 10 > covered,
        },
        charts: charts.len(),
        painted: painted.len(),
        coverage: covered as f32 / (side * side) as f32,
    })
}
