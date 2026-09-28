//! From decoded voxels to a triangle mesh (o-voxel's `flexible_dual_grid_to_mesh`):
//! each voxel holds one vertex inside it, and each voxel edge the surface
//! crosses joins the four voxels around it with a quad, split along the
//! diagonal its split weights favour. Then the mesh is compacted, given normals,
//! and (optionally) coloured from the texture voxels.
use std::collections::HashMap;

pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    /// RGBA in [0, 1], one per vertex.
    pub colors: Option<Vec<[f32; 4]>>,
    /// Metallic and roughness per vertex.
    pub metal_rough: Option<Vec<[f32; 2]>>,
}

fn key(c: [i32; 3]) -> u64 {
    ((c[0] as u64 & 0x1f_ffff) << 42) | ((c[1] as u64 & 0x1f_ffff) << 21) | (c[2] as u64 & 0x1f_ffff)
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// The shape decoder's 7 values per voxel at `res` (the voxels' coordinates) to a mesh in [-0.5, 0.5]³.
pub fn dual_grid(coords: &[[i32; 3]], values: &[f32], res: usize) -> Mesh {
    let n = coords.len();
    let voxel = 1. / res as f32;
    let margin = 0.5f32;
    let mut positions = Vec::with_capacity(n);
    let mut split = Vec::with_capacity(n);
    let mut index: HashMap<u64, u32> = HashMap::with_capacity(n * 2);
    for (i, c) in coords.iter().enumerate() {
        let v = &values[i * 7..i * 7 + 7];
        let d = |k: usize| (1. + 2. * margin) * sigmoid(v[k]) - margin;
        positions.push([(c[0] as f32 + d(0)) * voxel - 0.5, (c[1] as f32 + d(1)) * voxel - 0.5, (c[2] as f32 + d(2)) * voxel - 0.5]);
        split.push(softplus(v[6]));
        index.insert(key(*c), i as u32);
    }
    // The voxels around an edge along x, y and z (o-voxel's edge_neighbor_voxel_offset).
    const AROUND: [[[i32; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let mut triangles = Vec::new();
    for (i, c) in coords.iter().enumerate() {
        for axis in 0..3 {
            if values[i * 7 + 3 + axis] <= 0. {
                continue;
            }
            let mut q = [0u32; 4];
            let mut whole = true;
            for (k, o) in AROUND[axis].iter().enumerate() {
                match index.get(&key([c[0] + o[0], c[1] + o[1], c[2] + o[2]])) {
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
            let w02 = split[q[0] as usize] * split[q[2] as usize];
            let w13 = split[q[1] as usize] * split[q[3] as usize];
            if w02 > w13 {
                triangles.push([q[0], q[1], q[2]]);
                triangles.push([q[0], q[2], q[3]]);
            } else {
                triangles.push([q[0], q[1], q[3]]);
                triangles.push([q[3], q[1], q[2]]);
            }
        }
    }
    let mut mesh = Mesh { positions, triangles, colors: None, metal_rough: None };
    mesh.compact();
    mesh
}

impl Mesh {
    /// Drops vertices no triangle uses, and degenerate triangles.
    pub fn compact(&mut self) {
        self.triangles.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
        let mut remap = vec![u32::MAX; self.positions.len()];
        let mut positions = Vec::new();
        let mut colors = self.colors.as_ref().map(|_| Vec::new());
        let mut mr = self.metal_rough.as_ref().map(|_| Vec::new());
        for t in &mut self.triangles {
            for v in t.iter_mut() {
                let old = *v as usize;
                if remap[old] == u32::MAX {
                    remap[old] = positions.len() as u32;
                    positions.push(self.positions[old]);
                    if let (Some(out), Some(src)) = (colors.as_mut(), self.colors.as_ref()) {
                        out.push(src[old]);
                    }
                    if let (Some(out), Some(src)) = (mr.as_mut(), self.metal_rough.as_ref()) {
                        out.push(src[old]);
                    }
                }
                *v = remap[old];
            }
        }
        self.positions = positions;
        self.colors = colors;
        self.metal_rough = mr;
    }

    /// Area-weighted vertex normals.
    pub fn normals(&self) -> Vec<[f32; 3]> {
        let mut n = vec![[0f32; 3]; self.positions.len()];
        for t in &self.triangles {
            let [a, b, c] = t.map(|i| self.positions[i as usize]);
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let f = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
            for &i in t {
                for k in 0..3 {
                    n[i as usize][k] += f[k];
                }
            }
        }
        for v in &mut n {
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if l > 0. {
                for x in v.iter_mut() {
                    *x /= l;
                }
            } else {
                *v = [0., 1., 0.];
            }
        }
        n
    }

    /// Colours each vertex from the texture voxels.
    pub fn color_from_voxels(&mut self, voxels: &Voxels) {
        let (colors, mr) = self.positions.iter().map(|&p| voxels.sample(p)).map(|a| ([a[0], a[1], a[2], a[5]], [a[3], a[4]])).unzip();
        self.colors = Some(colors);
        self.metal_rough = Some(mr);
    }
}

/// The texture voxels (6 values each: base colour, metallic, roughness, alpha, in
/// [0, 1]) on their `res`³ grid, for sampling anywhere near the surface.
pub struct Voxels<'a> {
    index: HashMap<u64, u32>,
    attrs: &'a [f32],
    res: usize,
}

impl<'a> Voxels<'a> {
    pub fn new(coords: &[[i32; 3]], attrs: &'a [f32], res: usize) -> Self {
        let mut index: HashMap<u64, u32> = HashMap::with_capacity(coords.len() * 2);
        for (i, c) in coords.iter().enumerate() {
            index.insert(key(*c), i as u32);
        }
        Self { index, attrs, res }
    }

    /// The values at `p` (in [-0.5, 0.5]³), clamped to [0, 1]: the voxels around it,
    /// trilinearly weighted, over those there are; off the voxels, the nearest within three.
    pub fn sample(&self, p: [f32; 3]) -> [f32; 6] {
        // Voxel centres sit at (index + 0.5) / res - 0.5.
        let g = p.map(|v| (v + 0.5) * self.res as f32 - 0.5);
        let base = g.map(|v| v.floor() as i32);
        let frac = [g[0] - base[0] as f32, g[1] - base[1] as f32, g[2] - base[2] as f32];
        let at = |j: u32| &self.attrs[j as usize * 6..j as usize * 6 + 6];
        let mut acc = [0f32; 6];
        let mut total = 0f32;
        for corner in 0..8 {
            let d = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            let Some(&j) = self.index.get(&key([base[0] + d[0], base[1] + d[1], base[2] + d[2]])) else { continue };
            let w: f32 = (0..3).map(|k| if d[k] == 1 { frac[k] } else { 1. - frac[k] }).product();
            for (a, v) in acc.iter_mut().zip(at(j)) {
                *a += w * v;
            }
            total += w;
        }
        if total > 1e-6 {
            acc = acc.map(|v| v / total);
        } else {
            let mut best: Option<(f32, u32)> = None;
            for dx in -3..=4 {
                for dy in -3..=4 {
                    for dz in -3..=4 {
                        let c = [base[0] + dx, base[1] + dy, base[2] + dz];
                        let Some(&j) = self.index.get(&key(c)) else { continue };
                        let d = (0..3).map(|k| (c[k] as f32 - g[k]).powi(2)).sum::<f32>();
                        if best.is_none_or(|(b, _)| d < b) {
                            best = Some((d, j));
                        }
                    }
                }
            }
            acc = match best {
                Some((_, j)) => at(j).try_into().unwrap(),
                None => [0.8, 0.8, 0.8, 0., 1., 1.],
            };
        }
        acc.map(|v| v.clamp(0., 1.))
    }
}
