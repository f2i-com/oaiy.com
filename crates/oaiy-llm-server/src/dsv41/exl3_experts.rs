//! Routed experts of an EXL3 checkpoint on the CPU ([`ExpertFormat::Exl3`](crate::dsv41::expert::ExpertFormat)): each
//! projection decoded from the record as it lies (`ggml_rs_wgpu::exl3::ProjAt`, the decoder EXL3's other models run
//! their host experts through), and between them the routed expert's own SwiGLU ([`swiglu`]: each sum rounded to bf16,
//! clamped at the limit, `silu(gate) * up` times the route weight, rounded again), its output rounded to bf16 as the fp4
//! path's is. Nothing of the fp8 path's activation quantization applies: the trellis has its own transforms, and rounds
//! where exllamav3 rounds.

use std::sync::Mutex;

use ggml_rs_wgpu::exl3::ProjAt;
use oaiy_engine::backend::parallel_rows;
use oaiy_engine::{Error, Result};

use crate::dsv41::expert::{swiglu, Exl3Proj, DIM};
use crate::dsv41::formats::to_bf16;

/// The codebook multiplier each projection of a record must carry (mul1's): the only codebook decoded here.
const MUL1: [u8; 4] = 0x83dc_d12d_u32.to_le_bytes();

/// Tile columns (16 outputs each) a task of a decode step's projections: 9 a gate's or an up's, 20 a down's, so a
/// token's six experts are some 230 tasks for the cores.
const STEP_COLS: usize = 16;

fn at(p: &Exl3Proj) -> ProjAt {
    ProjAt { suh: p.suh, svh: p.svh, mul1: p.mul1, trellis: p.trellis, k: p.k, n: p.n, tile_words: p.tile_words }
}

/// Whether `record` holds a layer's experts as `layout` says: each projection's codebook mul1's, and its parts within
/// the record. The decoder takes this for granted, so a record read from a damaged file is refused here.
pub fn check(record: &[u8], layout: &[Exl3Proj; 3]) -> Result<()> {
    for p in layout {
        let end = p.trellis + p.k / 16 * (p.n / 16) * p.tile_words * 2;
        if record.get(p.mul1..p.mul1 + 4) != Some(&MUL1[..]) || end > record.len() || p.suh + 2 * p.k > record.len() || p.svh + 2 * p.n > record.len() {
            return Err(Error::Format(format!("an EXL3 expert record ({} bytes) does not hold its {}x{} projection as its layer's layout says", record.len(), p.k, p.n)));
        }
    }
    Ok(())
}

/// One expert's rows (`x`, `[m, DIM]`) through its `record`: gate and up, the SwiGLU, then down, on this thread (a
/// prompt's experts run many at once, one a worker): `[m, DIM]`, rounded to bf16.
pub fn forward_rows(record: &[u8], layout: &[Exl3Proj; 3], x: &[f32], weights: Option<&[f32]>, limit: f32) -> Result<Vec<f32>> {
    check(record, layout)?;
    let m = x.len() / DIM;
    let [g, u, d] = layout.map(|p| at(&p));
    let (gate, up) = (g.forward_host(record, x, m), u.forward_host(record, x, m));
    let h = swiglu(&gate, &up, weights, limit);
    Ok(d.forward_host(record, &h, m).into_iter().map(to_bf16).collect())
}

/// A decode step's experts (one row, `x`), each `records[i]` with its route weight `weights[i]`, at once on the host's
/// cores: each projection's tile columns in runs that are tasks of their own, a token's six experts being too
/// few tasks for the cores alone. Each expert's output, `[DIM]`, rounded to bf16: the same as [`forward_rows`] gives
/// it, whatever the runs.
pub fn forward_step(records: &[&[u8]], layout: &[Exl3Proj; 3], x: &[f32], weights: &[f32], limit: f32) -> Result<Vec<Vec<f32>>> {
    for r in records {
        check(r, layout)?;
    }
    let [g, u, d] = layout.map(|p| at(&p));
    // each projection of every expert: its rows prepared, its columns' sums in runs, put together and finished
    let run = |p: &ProjAt, xs: &[Vec<f32>]| -> Vec<Vec<f32>> {
        let ntiles = p.n / 16;
        let tasks: Vec<(usize, usize)> = (0..records.len()).flat_map(|i| (0..ntiles).step_by(STEP_COLS).map(move |c| (i, c))).collect();
        let sums: Vec<Vec<f32>> = each(tasks.len(), &|t| {
            let (i, c) = tasks[t];
            p.columns(records[i], &xs[i], 1, c..(c + STEP_COLS).min(ntiles))
        });
        let mut raw = vec![vec![0f32; p.n]; records.len()];
        for (&(i, c), s) in tasks.iter().zip(&sums) {
            raw[i][c * 16..c * 16 + s.len()].copy_from_slice(s);
        }
        let raw: Vec<Mutex<Vec<f32>>> = raw.into_iter().map(Mutex::new).collect();
        each(records.len(), &|i| p.finish(records[i], std::mem::take(&mut *raw[i].lock().unwrap_or_else(|e| e.into_inner())), 1))
    };
    let xg: Vec<Vec<f32>> = each(records.len(), &|i| g.prepare(records[i], x, 1));
    let xu: Vec<Vec<f32>> = each(records.len(), &|i| u.prepare(records[i], x, 1));
    let (gate, up) = (run(&g, &xg), run(&u, &xu));
    let h: Vec<Vec<f32>> = (0..records.len()).map(|i| swiglu(&gate[i], &up[i], Some(&weights[i..i + 1]), limit)).collect();
    let xd: Vec<Vec<f32>> = each(records.len(), &|i| d.prepare(records[i], &h[i], 1));
    Ok(run(&d, &xd).into_iter().map(|y| y.into_iter().map(to_bf16).collect()).collect())
}

/// `f(i)` for every `i` below `n` on the host's cores (`parallel_rows`), in order.
fn each<T: Send>(n: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    let out: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    parallel_rows(n, 1, &|b, e| {
        for i in b..e {
            *out[i].lock().unwrap_or_else(|p| p.into_inner()) = Some(f(i));
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap_or_else(|p| p.into_inner()).expect("every task made")).collect()
}
