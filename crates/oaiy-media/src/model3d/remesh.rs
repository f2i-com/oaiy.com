//! A clean surface for simplifying (CuMesh's `remesh_narrow_band_dc`, as TRELLIS.2's
//! GLB export runs it): the decoded dual-grid mesh has open and non-manifold edges
//! that stop edge collapses, so it is rebuilt as the level set one voxel out from
//! it (a thin closed shell) by dual contouring in a narrow band. Unlike CuMesh,
//! parts closed at the band's scale are filled first, so their shell has no inner
//! sheet.
//!
//! The unsigned distance to the mesh is exact near the surface: the triangles are
//! listed by the 2-voxel cells they overlap, and a query looks at the 27 cells
//! around it. The band's voxels are found from the cells the triangles reach,
//! dilated, rather than a coarse-to-fine BVH search. Each voxel's vertex is the
//! mean of its edges' crossings, and each crossed edge joins the four voxels
//! around it with a quad, split along the flatter diagonal.
use super::mesh::Mesh;
use std::collections::HashMap;

fn key(c: [i32; 3]) -> u64 {
    ((c[0] as u64 & 0x1f_ffff) << 42) | ((c[1] as u64 & 0x1f_ffff) << 21) | (c[2] as u64 & 0x1f_ffff)
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

/// The point of the triangle (a, b, c) closest to `p` (Ericson's).
fn closest_on_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0. && d2 <= 0. {
        a
    } else {
        let bp = sub(p, b);
        let (d3, d4) = (dot(ab, bp), dot(ac, bp));
        if d3 >= 0. && d4 <= d3 {
            b
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0. && d1 >= 0. && d3 <= 0. {
                let v = d1 / (d1 - d3);
                [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]]
            } else {
                let cp = sub(p, c);
                let (d5, d6) = (dot(ab, cp), dot(ac, cp));
                if d6 >= 0. && d5 <= d6 {
                    c
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0. && d2 >= 0. && d6 <= 0. {
                        let w = d2 / (d2 - d6);
                        [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]]
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0. && (d4 - d3) >= 0. && (d5 - d6) >= 0. {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            let bc = sub(c, b);
                            [b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]]
                        } else {
                            let denom = 1. / (va + vb + vc);
                            let (v, w) = (vb * denom, vc * denom);
                            [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]
                        }
                    }
                }
            }
        }
    }
}

/// The mesh's triangles by the `CELL`-sized cells their bounding boxes overlap (grid units).
struct Cells {
    index: HashMap<u64, (u32, u32)>,
    triangles: Vec<u32>,
}

const CELL: f32 = 2.;
/// How far the distance is exact; beyond it a query reports `FAR`.
const REACH: f32 = 2.;
const FAR: f32 = 3.;

impl Cells {
    fn new(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Self {
        let span = |t: &[u32; 3]| {
            let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
            for &i in t {
                for k in 0..3 {
                    let c = (verts[i as usize][k] / CELL).floor() as i32;
                    lo[k] = lo[k].min(c);
                    hi[k] = hi[k].max(c);
                }
            }
            (lo, hi)
        };
        let mut counts: HashMap<u64, u32> = HashMap::new();
        for t in tris {
            let (lo, hi) = span(t);
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        *counts.entry(key([x, y, z])).or_default() += 1;
                    }
                }
            }
        }
        let mut index: HashMap<u64, (u32, u32)> = HashMap::with_capacity(counts.len());
        let mut at = 0u32;
        for (k, n) in &counts {
            index.insert(*k, (at, 0));
            at += n;
        }
        let mut triangles = vec![0u32; at as usize];
        for (ti, t) in tris.iter().enumerate() {
            let (lo, hi) = span(t);
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        let e = index.get_mut(&key([x, y, z])).unwrap();
                        triangles[(e.0 + e.1) as usize] = ti as u32;
                        e.1 += 1;
                    }
                }
            }
        }
        Self { index, triangles }
    }

    /// The mesh's point closest to `p` and its squared distance, when one is within
    /// `reach` (every triangle that near is looked at).
    fn nearest(&self, p: [f32; 3], reach: f32, verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Option<([f32; 3], f32)> {
        let lo = p.map(|v| ((v - reach) / CELL).floor() as i32);
        let hi = p.map(|v| ((v + reach) / CELL).floor() as i32);
        let mut best: Option<([f32; 3], f32)> = None;
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let Some(&(start, n)) = self.index.get(&key([x, y, z])) else { continue };
                    for &ti in &self.triangles[start as usize..(start + n) as usize] {
                        let t = tris[ti as usize];
                        let c = closest_on_triangle(p, verts[t[0] as usize], verts[t[1] as usize], verts[t[2] as usize]);
                        let d = sub(p, c);
                        let d = dot(d, d);
                        if best.is_none_or(|(_, b)| d < b) {
                            best = Some((c, d));
                        }
                    }
                }
            }
        }
        best.filter(|(_, d)| *d <= reach * reach)
    }
}

/// The decoded mesh on the remeshing grid: `resolution`³ voxels over the cube widened
/// by three voxels (as TRELLIS.2's), with its triangles listed by cell for exact
/// distances and closest points near it.
pub struct Surface {
    res: f32,
    scale: f32,
    /// Grid units: vertex v of the grid sits at (v / res - 0.5) · scale.
    verts: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    cells: Cells,
}

impl Surface {
    pub fn new(mesh: &Mesh, resolution: usize) -> Self {
        let res = resolution as f32;
        let scale = (res + 3.) / res;
        let verts: Vec<[f32; 3]> = mesh.positions.iter().map(|p| p.map(|v| (v / scale + 0.5) * res)).collect();
        let cells = Cells::new(&verts, &mesh.triangles);
        Self { res, scale, verts, tris: mesh.triangles.clone(), cells }
    }

    fn to_world(&self, g: [f32; 3]) -> [f32; 3] {
        g.map(|v| (v / self.res - 0.5) * self.scale)
    }

    /// The unsigned distance from `g` (grid units) to the mesh, exact up to `REACH`, else `FAR`.
    fn distance(&self, g: [f32; 3]) -> f32 {
        self.cells.nearest(g, REACH, &self.verts, &self.tris).map_or(FAR, |(_, d)| d.sqrt())
    }

    /// The mesh's point closest to `p` (in [-0.5, 0.5]³), when one is within `reach` voxels.
    pub fn closest(&self, p: [f32; 3], reach: f32) -> Option<[f32; 3]> {
        let g = p.map(|v| (v / self.scale + 0.5) * self.res);
        self.cells.nearest(g, reach, &self.verts, &self.tris).map(|(c, _)| self.to_world(c))
    }
}

/// Runs `f` over `items` on all cores, in order.
pub(super) fn par_map<T: Sync, R: Send + Default + Clone>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(64);
    let chunk = items.len().div_ceil(threads).max(1);
    let mut out = vec![R::default(); items.len()];
    std::thread::scope(|s| {
        for (src, dst) in items.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let f = &f;
            s.spawn(move || {
                for (i, item) in src.iter().enumerate() {
                    dst[i] = f(item);
                }
            });
        }
    });
    out
}

/// The shell one voxel out from the surface, dual-contoured on its grid. Where the
/// surface is closed, only the shell's outer sheet is kept (see `seal`).
pub fn remesh(surface: &Surface) -> Mesh {
    let resolution = surface.res as usize;
    let band = 1f32;
    let udf = |p: [f32; 3]| surface.distance(p);

    // Candidate voxels: the cells the triangles reach, dilated by one cell, split in eight.
    let mut near: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let unkey = |k: u64| [((k >> 42) & 0x1f_ffff) as i32, ((k >> 21) & 0x1f_ffff) as i32, (k & 0x1f_ffff) as i32];
    for &k in surface.cells.index.keys() {
        let c = unkey(k);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    near.insert(key([c[0] + dx, c[1] + dy, c[2] + dz]));
                }
            }
        }
    }
    let mut candidates: Vec<[i32; 3]> = Vec::with_capacity(near.len() * 8);
    for k in near {
        let c = unkey(k);
        for s in 0..8 {
            let v = [c[0] * 2 + (s & 1), c[1] * 2 + ((s >> 1) & 1), c[2] * 2 + ((s >> 2) & 1)];
            if v.iter().all(|&x| x >= 0 && (x as usize) < resolution) {
                candidates.push(v);
            }
        }
    }
    // The band: voxels whose centre is within √3/2 of the shell.
    let keep = par_map(&candidates, |c| (udf([c[0] as f32 + 0.5, c[1] as f32 + 0.5, c[2] as f32 + 0.5]) - band).abs() < 0.87);
    let mut voxels: Vec<[i32; 3]> = candidates.iter().zip(&keep).filter(|(_, k)| **k).map(|(c, _)| *c).collect();
    drop(candidates);
    voxels.sort_unstable();
    let voxel_index: HashMap<u64, u32> = voxels.iter().enumerate().map(|(i, c)| (key(*c), i as u32)).collect();
    // The level set's values at the voxels' corners.
    let mut corner_index: HashMap<u64, u32> = HashMap::with_capacity(voxels.len() * 2);
    let mut corners: Vec<[i32; 3]> = Vec::new();
    for c in &voxels {
        for s in 0..8 {
            let v = [c[0] + (s & 1), c[1] + ((s >> 1) & 1), c[2] + ((s >> 2) & 1)];
            corner_index.entry(key(v)).or_insert_with(|| {
                corners.push(v);
                (corners.len() - 1) as u32
            });
        }
    }
    let mut values = par_map(&corners, |v| udf([v[0] as f32, v[1] as f32, v[2] as f32]) - band);
    seal(&corners, &corner_index, &mut values);
    let value = |x: i32, y: i32, z: i32| values[corner_index[&key([x, y, z])] as usize];
    // Dual vertices and the crossed edges at each voxel's (+1, +1) corner, as CuMesh's kernel.
    struct Dual {
        vertex: [f32; 3],
        crossed: [i8; 3],
    }
    let duals: Vec<Dual> = voxels
        .iter()
        .map(|&[vx, vy, vz]| {
            let mut sum = [0f32; 3];
            let mut count = 0;
            let mut crossed = [0i8; 3];
            for axis in 0..3 {
                for u in 0..=1 {
                    for v in 0..=1 {
                        let (p1, p2) = match axis {
                            0 => ([vx, vy + u, vz + v], [vx + 1, vy + u, vz + v]),
                            1 => ([vx + u, vy, vz + v], [vx + u, vy + 1, vz + v]),
                            _ => ([vx + u, vy + v, vz], [vx + u, vy + v, vz + 1]),
                        };
                        let (a, b) = (value(p1[0], p1[1], p1[2]), value(p2[0], p2[1], p2[2]));
                        let changes = (a < 0. && b >= 0.) || (a >= 0. && b < 0.);
                        if changes {
                            let t = -a / (b - a);
                            for k in 0..3 {
                                sum[k] += p1[k] as f32 + t * (p2[k] - p1[k]) as f32;
                            }
                            count += 1;
                        }
                        if u == 1 && v == 1 {
                            crossed[axis] = if a < 0. && b >= 0. { 1 } else if a >= 0. && b < 0. { -1 } else { 0 };
                        }
                    }
                }
            }
            let vertex = if count > 0 { sum.map(|s| s / count as f32) } else { [vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5] };
            Dual { vertex, crossed }
        })
        .collect();
    const AROUND: [[[i32; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let positions: Vec<[f32; 3]> = duals.iter().map(|d| surface.to_world(d.vertex)).collect();
    let normal = |t: [u32; 3]| {
        let [a, b, c] = t.map(|i| positions[i as usize]);
        cross(sub(b, a), sub(c, a))
    };
    let mut triangles = Vec::new();
    for (i, d) in duals.iter().enumerate() {
        let c = voxels[i];
        for axis in 0..3 {
            if d.crossed[axis] == 0 {
                continue;
            }
            let mut q = [0u32; 4];
            let mut whole = true;
            for (k, o) in AROUND[axis].iter().enumerate() {
                match voxel_index.get(&key([c[0] + o[0], c[1] + o[1], c[2] + o[2]])) {
                    Some(&j) => q[k] = j,
                    None => {
                        whole = false;
                        break;
                    }
                }
            }
            if !whole {
                continue;
            }
            // CuMesh's quad_split_{1,2}_{p,n}: the winding follows the crossing's direction.
            let (s1, s2): ([usize; 6], [usize; 6]) = if d.crossed[axis] == 1 { ([0, 2, 1, 0, 3, 2], [0, 3, 1, 3, 2, 1]) } else { ([0, 1, 2, 0, 2, 3], [0, 1, 3, 3, 1, 2]) };
            let tri = |s: &[usize; 6], k: usize| [q[s[k * 3]], q[s[k * 3 + 1]], q[s[k * 3 + 2]]];
            let align = |s: &[usize; 6]| dot(normal(tri(s, 0)), normal(tri(s, 1))).abs();
            let s = if align(&s1) > align(&s2) { s1 } else { s2 };
            triangles.push(tri(&s, 0));
            triangles.push(tri(&s, 1));
        }
    }
    let mut out = Mesh { positions, triangles, colors: None, metal_rough: None };
    out.compact();
    out
}

/// Fills closed parts, so their shell's inner sheet is not contoured. The corners
/// outside the band's slab (value ≥ 0) fall into layers: one around the outside of
/// each part, and one inside each closed part (or hollow). A layer is an inner one
/// when, from each of its six extreme corners, the next corner outward is in the
/// slab; the outside layer's extremes look out past the band instead. An inner
/// layer's corners are made solid. A part with a gap wider than the band joins its
/// layers into one, which is left as it is: an open surface keeps both sheets (as
/// the decoded surface of a whole object usually does, and TRELLIS.2's shell always).
fn seal(corners: &[[i32; 3]], index: &HashMap<u64, u32>, values: &mut [f32]) {
    const STEPS: [[i32; 3]; 6] = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
    let n = corners.len();
    let mut layer = vec![u32::MAX; n];
    let mut stack = Vec::new();
    let mut members = Vec::new();
    for start in 0..n {
        if values[start] < 0. || layer[start] != u32::MAX {
            continue;
        }
        layer[start] = start as u32;
        stack.push(start as u32);
        members.clear();
        // The extreme corners along -x, +x, -y, +y, -z, +z.
        let mut extreme = [start as u32; 6];
        while let Some(i) = stack.pop() {
            members.push(i);
            let c = corners[i as usize];
            for (k, e) in extreme.iter_mut().enumerate() {
                let (axis, sign) = (k / 2, if k % 2 == 0 { -1 } else { 1 });
                if (c[axis] - corners[*e as usize][axis]) * sign > 0 {
                    *e = i;
                }
            }
            for s in STEPS {
                let Some(&j) = index.get(&key([c[0] + s[0], c[1] + s[1], c[2] + s[2]])) else { continue };
                if values[j as usize] >= 0. && layer[j as usize] == u32::MAX {
                    layer[j as usize] = start as u32;
                    stack.push(j);
                }
            }
        }
        let inner = extreme.iter().zip(STEPS).all(|(&e, s)| {
            let c = corners[e as usize];
            index.get(&key([c[0] + s[0], c[1] + s[1], c[2] + s[2]])).is_some_and(|&j| values[j as usize] < 0.)
        });
        if inner {
            for &i in &members {
                values[i as usize] = -1.;
            }
        }
    }
}
