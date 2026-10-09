//! A MoE layer's experts as their GGUF holds them (`QuantExpertsData`), and the layer on the host
//! (`QuantMoeCpu`), which the kernels are checked against.

use super::*;

/// One MoE layer's experts as a GGUF holds them.
pub struct QuantExpertsData {
    pub hidden: usize,
    pub ff: usize,
    /// The routed experts.
    pub experts: usize,
    /// Routed gate and up: `[experts][ff rows][hidden cols]`; down: `[experts][hidden rows][ff cols]`. Raw block
    /// bytes, row-major, each row a whole number of blocks (a GGUF's `ffn_gate_exps` of 2560 x 640 x 512 and so on).
    pub gate: (GgmlType, Vec<u8>),
    pub up: (GgmlType, Vec<u8>),
    pub down: (GgmlType, Vec<u8>),
    /// The shared expert, dequantised by the loader: gate and up `[ff, hidden]`, down `[hidden, ff]`, f32 row-major.
    pub shared: [Vec<f32>; 3],
}

/// The bytes of a row of `cols` weights of type `t`.
pub(super) fn row_bytes(t: GgmlType, cols: usize) -> usize {
    cols / t.block_size() * t.type_size()
}

impl QuantExpertsData {
    /// Each tensor's bytes are its shape's (a row a whole number of its type's blocks), of a type `ggml_quants` decodes.
    pub fn validate(&self) -> Result<(), String> {
        let (h, f, e) = (self.hidden, self.ff, self.experts);
        if h == 0 || f == 0 || e == 0 {
            return Err("experts of no size".into());
        }
        for (what, (t, bytes), rows, cols) in [("gate", &self.gate, f, h), ("up", &self.up, f, h), ("down", &self.down, h, f)] {
            if !ggml_quants::is_supported(*t) {
                return Err(format!("the experts' {what}: no decoder for {}", t.name()));
            }
            if cols % t.block_size() != 0 {
                return Err(format!("the experts' {what}: rows of {cols} are not whole {} blocks", t.name()));
            }
            let want = e * rows * row_bytes(*t, cols);
            if bytes.len() != want {
                return Err(format!("the experts' {what}: {} bytes, {want} for {e} of {rows} x {cols} in {}", bytes.len(), t.name()));
            }
        }
        for (what, m) in ["gate", "up", "down"].iter().zip(&self.shared) {
            if m.len() != h * f {
                return Err(format!("the shared expert's {what}: {} values, not {}", m.len(), h * f));
            }
        }
        Ok(())
    }
}

/// `y = W x`, `W` `[n, k]` row-major, summed in f64 (the reference's).
fn matvec(w: &[f32], k: usize, x: &[f32], y: &mut [f32]) {
    for (row, out) in w.chunks_exact(k).zip(y.iter_mut()) {
        *out = row.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// The experts on the host: the reference the GPU's are held against, and where they run without a GPU (or past its
/// budget). Each expert a call's rows share is dequantised once for them.
pub struct QuantMoeCpu {
    data: QuantExpertsData,
}

impl std::fmt::Debug for QuantMoeCpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantMoeCpu({} routed experts of {}x{} in {}, and the shared one, on the host)", self.data.experts, self.data.hidden, self.data.ff, self.data.gate.0.name())
    }
}

impl QuantMoeCpu {
    /// Matrix `e` of a tensor of `rows` by `cols` matrices, dequantised.
    fn matrix((t, bytes): &(GgmlType, Vec<u8>), e: usize, rows: usize, cols: usize) -> Vec<f32> {
        let each = rows * row_bytes(*t, cols);
        let mut out = vec![0f32; rows * cols];
        ggml_quants::dequantize(*t, &bytes[e * each..(e + 1) * each], &mut out).expect("a validated tensor");
        out
    }
}

impl Experts for QuantMoeCpu {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let d = &self.data;
        let (h, f, n) = (d.hidden, d.ff, d.experts);
        let x = x.to_host();
        let logits = logits.to_host();
        let (xs, ls) = (x.data(), logits.data());
        let rows = xs.len() / h;
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * (n + 1)..(r + 1) * (n + 1)], top_k)).collect();
        // each routed expert's rows (and their weights)
        let mut by: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for (r, a) in assign.iter().enumerate() {
            for &(e, w) in &a[..a.len() - 1] {
                by[e].push((r, w));
            }
        }
        let expert = |g: &[f32], u: &[f32], dn: &[f32], xr: &[f32], w: f32| -> Vec<f32> {
            let (mut a, mut b, mut y) = (vec![0f32; f], vec![0f32; f], vec![0f32; h]);
            matvec(g, h, xr, &mut a);
            matvec(u, h, xr, &mut b);
            for (a, b) in a.iter_mut().zip(&b) {
                *a = silu(*a) * b;
            }
            matvec(dn, f, &a, &mut y);
            y.iter_mut().for_each(|v| *v *= w);
            y
        };
        let routed: Vec<(usize, Vec<f32>)> = by
            .par_iter()
            .enumerate()
            .filter(|(_, rs)| !rs.is_empty())
            .flat_map_iter(|(e, rs)| {
                let (g, u, dn) = (Self::matrix(&d.gate, e, f, h), Self::matrix(&d.up, e, f, h), Self::matrix(&d.down, e, h, f));
                rs.iter().map(|&(r, w)| (r, expert(&g, &u, &dn, &xs[r * h..(r + 1) * h], w))).collect::<Vec<_>>()
            })
            .collect();
        let mut out = vec![0f32; rows * h];
        for (r, y) in routed {
            out[r * h..(r + 1) * h].iter_mut().zip(&y).for_each(|(o, v)| *o += v);
        }
        // the shared expert on every row, weighted by its gate's sigmoid
        out.par_chunks_mut(h).enumerate().for_each(|(r, o)| {
            let y = expert(&d.shared[0], &d.shared[1], &d.shared[2], &xs[r * h..(r + 1) * h], assign[r].last().expect("the shared expert").1);
            o.iter_mut().zip(&y).for_each(|(o, v)| *o += v);
        });
        Tensor::from_vec(out, vec![rows, h])
    }

    fn on_host(&self) -> bool {
        true
    }
}

/// The experts of `data` on the host, the reference: [`crate::quant_host::quant_experts_host`] is what a model runs.
pub fn quant_experts_cpu(data: QuantExpertsData) -> Result<Box<dyn Experts>, String> {
    data.validate()?;
    Ok(Box::new(QuantMoeCpu { data }))
}
