//! Pixal3D's view-aligned conditioning: each voxel's centre projected into the
//! image through the camera it was rendered with (Blender's convention, TRELLIS's
//! front view), and the image's feature maps sampled there (bilinear, border
//! padding, `align_corners=False`, as `F.grid_sample`).
use candle_core::{Device, Result, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Horizontal field of view, radians.
    pub fov: f64,
    pub distance: f64,
    pub mesh_scale: f64,
}

impl Camera {
    /// The distance that frames the unit cube at `fov` (inference.py's `distance_from_fov`):
    /// the grid's point (-1, 0, 0) lands on the image's left edge.
    pub fn framing(fov: f64, mesh_scale: f64, image_res: f64, extend_pixel: f64) -> Self {
        let f = 16. / (fov / 2.).tan() * image_res / 32.;
        // The grid point (-1, 0, 0), rotated (x, -z, y) and scaled.
        let (xw, yw) = (-1. / mesh_scale / 2., 0.);
        let xt = 0. - extend_pixel;
        let x_ndc = xt - image_res / 2.;
        Self { fov, distance: f * xw / x_ndc - yw, mesh_scale }
    }

    /// A grid point's pixel position in an `image_res` image. `g`: the point in [-1, 1]³.
    pub fn project(&self, g: [f64; 3], image_res: f64) -> (f64, f64) {
        // grid @ R^T with R = [[1,0,0],[0,0,-1],[0,1,0]], then /scale/2; the camera
        // sits at (0, -distance, 0) looking along +y (Blender: -z in camera space).
        let s = 2. * self.mesh_scale;
        let (x_cam, y_cam, z_cam) = (g[0] / s, g[1] / s, g[2] / s - self.distance);
        let f = 16. / (self.fov / 2.).tan() * image_res / 32.;
        let x = f * x_cam / (-z_cam + 1e-8) + image_res / 2.;
        let y = -(f * y_cam / (-z_cam + 1e-8)) + image_res / 2.;
        (x, y)
    }
}

/// Where a grid index sits in [-1, 1] (`torch.linspace(-1, 1, res)`).
pub fn grid_point(c: [i32; 3], res: usize) -> [f64; 3] {
    let t = |v: i32| if res > 1 { -1. + 2. * v as f64 / (res - 1) as f64 } else { 0. };
    [t(c[0]), t(c[1]), t(c[2])]
}

/// Bilinear taps into a `w`×`h` map for a point at pixel (x, y) of an `image_res` image:
/// four (row index, weight) pairs, border-clamped as grid_sample does.
pub fn taps(x: f64, y: f64, image_res: f64, w: usize, h: usize) -> [(usize, f32); 4] {
    // The normalized coordinate grid_sample gets, back to the map's pixels (align_corners=False).
    let u = (x + 0.5) / image_res * 2. - 1.;
    let v = (y + 0.5) / image_res * 2. - 1.;
    let px = (((u + 1.) * w as f64 - 1.) / 2.).clamp(0., (w - 1) as f64);
    let py = (((v + 1.) * h as f64 - 1.) / 2.).clamp(0., (h - 1) as f64);
    let (x0, y0) = (px.floor() as usize, py.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = ((px - x0 as f64) as f32, (py - y0 as f64) as f32);
    [(y0 * w + x0, (1. - fx) * (1. - fy)), (y0 * w + x1, fx * (1. - fy)), (y1 * w + x0, (1. - fx) * fy), (y1 * w + x1, fx * fy)]
}

/// Samples rows of `map` [rows, C] with each point's four taps: [points, C].
pub fn sample(map: &Tensor, taps: &[[(usize, f32); 4]], dev: &Device) -> Result<Tensor> {
    let n = taps.len();
    let mut out: Option<Tensor> = None;
    for k in 0..4 {
        let ids: Vec<u32> = taps.iter().map(|t| t[k].0 as u32).collect();
        let ws: Vec<f32> = taps.iter().map(|t| t[k].1).collect();
        let rows = map.index_select(&Tensor::from_vec(ids, n, dev)?, 0)?;
        let part = rows.broadcast_mul(&Tensor::from_vec(ws, (n, 1), dev)?)?;
        out = Some(match out {
            Some(o) => (o + part)?,
            None => part,
        });
    }
    Ok(out.expect("four taps"))
}
