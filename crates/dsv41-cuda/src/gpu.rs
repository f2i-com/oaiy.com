//! One GPU: context, stream, the compiled kernels, and a typed wrapper per
//! kernel.
//!
//! # Unsafe boundary
//!
//! cudarc marks a kernel launch `unsafe` because it cannot check the
//! argument list against the kernel's C signature. Every launch in this
//! crate goes through one wrapper here whose argument list is written next
//! to (and must be kept in sync with) the kernel in `kernels.cu`; buffer
//! lengths are checked with `assert!` before the launch so a kernel can
//! never index past an allocation it was handed. Nothing else in the crate
//! is `unsafe`.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use nrob::{Error, Result};

const SRC: &str = include_str!("kernels.cu");
const KERNELS: &[&str] = &[
    "act_quant_fp8",
    "act_quant_fp8_to",
    "act_quant_fp4",
    "gemv_fp8",
    "gemv_fp8_token",
    "gemv_bf16",
    "shared_gate_up",
    "gemv_fp8w",
    "gemv_f32",
    "gemv_fp4",
    "swiglu",
    "moe_gate_up",
    "moe_down",
    "moe_reduce",
    "moe_reduce_host",
    "publish",
    "gather_rows",
    "scatter_add_rows",
    "add_round",
    "round_bf16",
    "rmsnorm",
    "rope",
    "hc_project",
    "hc_mix",
    "hc_project_mix",
    "hc_pre",
    "hc_pre_norm",
    "kv_finish",
    "hc_post",
    "sparse_attn",
    "index_scores",
    "compress_pool",
    "engram_gate",
];

/// Map any CUDA error into the engine's error type.
pub(crate) fn cu<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| Error::Unsupported(format!("cuda: {e:?}")))
}

/// Where [`Gpu::rope`] takes each row's position from.
pub enum Pos<'a> {
    /// One position per row, uploaded.
    Rows(&'a CudaSlice<i32>),
    /// `base + row / per`: rows in groups of `per` sharing a position
    /// (heads of one token), consecutive groups consecutive positions.
    Linear { base: usize, per: usize },
}

pub struct Gpu {
    pub ordinal: usize,
    ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    funcs: HashMap<&'static str, CudaFunction>,
}

fn rows_cfg(rows: usize) -> LaunchConfig {
    // 8 warps per block, one warp per output row / quant block
    LaunchConfig { grid_dim: (rows.div_ceil(8) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn elems_cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (n.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

impl Gpu {
    /// Open device `ordinal` and compile the kernels (--fmad=false: the CPU
    /// reference never contracts a*b+c, so the GPU must not either).
    pub fn new(ordinal: usize) -> Result<Gpu> {
        let ctx = cu(CudaContext::new(ordinal))?;
        // SAFETY: cudarc's per-buffer event tracking exists to order work
        // across streams. Everything this crate does on a device — kernels,
        // uploads, downloads, device copies — is issued on that device's one
        // stream, so stream order already covers every read/write edge, and
        // buffers are only freed after the host has stopped issuing work that
        // uses them (drop order on the same stream). With tracking on, every
        // launch pays event record/wait driver calls per buffer argument
        // (~76 us vs ~15 us per launch on WDDM, measured in ggml-rs-cuda's
        // PERF-02), which at thousands of launches per token dominates decode.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        // compile for this device, so the kernels can use its instructions
        // (e.g. the sm_89+ fp8 -> f16 conversion); unknown ones keep NVRTC's default
        let arch = match cu(ctx.compute_capability())? {
            (12, 0) => Some("compute_120"),
            (10, 0) => Some("compute_100"),
            (9, 0) => Some("compute_90"),
            (8, 9) => Some("compute_89"),
            (8, 6) => Some("compute_86"),
            (8, 0) => Some("compute_80"),
            _ => None,
        };
        let opts = CompileOptions { fmad: Some(false), arch, ..Default::default() };
        let ptx = cu(compile_ptx_with_opts(SRC, opts))?;
        let module = cu(ctx.load_module(ptx))?;
        let mut funcs = HashMap::new();
        for &k in KERNELS {
            funcs.insert(k, cu(module.load_function(k))?);
        }
        Ok(Gpu { ordinal, ctx, stream, funcs })
    }

    fn f(&self, name: &'static str) -> &CudaFunction {
        &self.funcs[name]
    }

    pub fn upload<T: DeviceRepr>(&self, host: &[T]) -> Result<CudaSlice<T>> {
        cu(self.stream.clone_htod(host))
    }

    pub fn download<T: DeviceRepr + Default + Clone>(&self, dev: &CudaSlice<T>) -> Result<Vec<T>> {
        cu(self.stream.clone_dtoh(dev))
    }

    pub fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> Result<CudaSlice<T>> {
        cu(self.stream.alloc_zeros::<T>(n.max(1)))
    }

    /// An uninitialized buffer for a kernel output that is written in full
    /// before anything reads it: [`zeros`](Self::zeros) without the memset
    /// launch, which at thousands of intermediates per token is not free.
    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> Result<CudaSlice<T>> {
        // SAFETY: cudarc marks `alloc` unsafe because the memory starts
        // unset. `T: ValidAsZeroBits` restricts this to plain numeric types
        // (f32, u8, i32, u64) for which every bit pattern is a valid value, so
        // even a stray read of unwritten memory is a wrong number, never UB;
        // and every caller hands the buffer to a kernel that writes all of it.
        cu(unsafe { self.stream.alloc::<T>(n.max(1)) })
    }

    /// Overwrite `dst[..host.len()]` from the host.
    pub fn write<T: DeviceRepr>(&self, host: &[T], dst: &mut CudaViewMut<'_, T>) -> Result<()> {
        cu(self.stream.memcpy_htod(host, dst))
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn sync(&self) -> Result<()> {
        cu(self.stream.synchronize())
    }

    /// (free, total) device memory in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        cu(self.ctx.mem_get_info())
    }

    // ------------------------------------------------------------ quantizers

    /// formats::fake_quant_fp8 in place over all of `x` (len % 32 == 0).
    /// [`act_quant_fp8`](Self::act_quant_fp8) into a new buffer, `x` untouched.
    pub fn act_quant_fp8_to(&self, x: &CudaView<'_, f32>) -> Result<CudaSlice<f32>> {
        let n = x.len();
        assert_eq!(n % 32, 0);
        let mut y = self.alloc::<f32>(n)?;
        let ni = n as i32;
        let mut b = self.stream.launch_builder(self.f("act_quant_fp8_to"));
        b.arg(x).arg(&mut y).arg(&ni);
        // SAFETY: act_quant_fp8_to(const float* x, float* y, int n); one warp per 32 values, both n long.
        cu(unsafe { b.launch(rows_cfg(n / 32)) })?;
        Ok(y)
    }

    pub fn act_quant_fp8(&self, x: &mut CudaViewMut<'_, f32>) -> Result<()> {
        let n = x.len();
        assert_eq!(n % 32, 0);
        let ni = n as i32;
        let mut b = self.stream.launch_builder(self.f("act_quant_fp8"));
        b.arg(x).arg(&ni);
        // SAFETY: act_quant_fp8(float* x, int n); one warp per 32 values, n checked above.
        cu(unsafe { b.launch(rows_cfg(n / 32)) })?;
        Ok(())
    }

    /// attention's fp4 round trip in place: `block` 16 or 32; `e4m3_scale`
    /// selects the compressed-KV scale (else the indexer's e8m0).
    pub fn act_quant_fp4(&self, x: &mut CudaViewMut<'_, f32>, block: usize, e4m3_scale: bool) -> Result<()> {
        let n = x.len();
        assert!(block == 16 || block == 32);
        assert_eq!(n % block, 0);
        let (ni, bi, ki) = (n as i32, block as i32, i32::from(e4m3_scale));
        let mut b = self.stream.launch_builder(self.f("act_quant_fp4"));
        b.arg(x).arg(&ni).arg(&bi).arg(&ki);
        // SAFETY: act_quant_fp4(float* x, int n, int block, int kind); one warp per block.
        cu(unsafe { b.launch(rows_cfg(n / block)) })?;
        Ok(())
    }

    // ------------------------------------------------------------ GEMV

    /// fp8 weight `[n, k]` with 32x32 e8m0 tiles; `xq` already fp8-quantized `[nt, k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_fp8(
        &self,
        xq: &CudaView<'_, f32>,
        w: &CudaSlice<u8>,
        s: &CudaSlice<u8>,
        y: &mut CudaViewMut<'_, f32>,
        n: usize,
        k: usize,
        nt: usize,
        round: bool,
    ) -> Result<()> {
        assert!(w.len() >= n * k && s.len() >= n.div_ceil(32) * k.div_ceil(32) && y.len() >= n * nt);
        let (ni, ki, ti, ri) = (n as i32, k as i32, nt as i32, i32::from(round));
        let mut b = self.stream.launch_builder(self.f("gemv_fp8"));
        b.arg(xq).arg(w).arg(s).arg(y).arg(&ni).arg(&ki).arg(&ti).arg(&ri);
        // SAFETY: gemv_fp8(const float* x, const u8* w, const u8* s, float* y, int n, int k, int nt, int round);
        // x is read as [nt, k] (caller's buffer), w/s/y lengths checked above.
        cu(unsafe { b.launch(rows_cfg(n)) })?;
        Ok(())
    }

    /// The shared expert's gate/up and SwiGLU for one token: `h = bf16(swiglu(
    /// bf16(w1 . xq), bf16(w3 . xq)))`, what two [`gemv_fp8`](Self::gemv_fp8)s and a
    /// [`swiglu`](Self::swiglu) give, in one launch. `None` if the shape does
    /// not fit (k % 32, unaligned rows).
    #[allow(clippy::too_many_arguments)]
    pub fn shared_gate_up(
        &self,
        xq: &CudaView<'_, f32>,
        w1: &CudaSlice<u8>,
        s1: &CudaSlice<u8>,
        w3: &CudaSlice<u8>,
        s3: &CudaSlice<u8>,
        n: usize,
        k: usize,
        lim: f32,
    ) -> Result<Option<CudaSlice<f32>>> {
        if !k.is_multiple_of(32) || !self.addr(&w1.as_view()).is_multiple_of(16) || !self.addr(&w3.as_view()).is_multiple_of(16) {
            return Ok(None);
        }
        let sn = n.div_ceil(32) * (k / 32);
        assert!(xq.len() >= k && w1.len() >= n * k && w3.len() >= n * k && s1.len() >= sn && s3.len() >= sn);
        let mut h = self.alloc::<f32>(n)?;
        let (ni, ki) = (n as i32, k as i32);
        let mut b = self.stream.launch_builder(self.f("shared_gate_up"));
        b.arg(xq).arg(w1).arg(s1).arg(w3).arg(s3).arg(&mut h).arg(&ni).arg(&ki).arg(&lim);
        // SAFETY: shared_gate_up(const float* xq, const u8* w1, const u8* s1, const u8* w3, const u8* s3,
        // float* h, int n, int k, float lim); lengths asserted, rows 16-byte aligned (bases checked,
        // k % 32 == 0); one warp per row.
        cu(unsafe { b.launch(rows_cfg(n)) })?;
        Ok(Some(h))
    }

    /// One-token fp8 GEMV, `y = bf16?(W . quant?(x))`: [`act_quant_fp8`](Self::act_quant_fp8)
    /// (when `quant`) and [`gemv_fp8`](Self::gemv_fp8) in one launch, bit for bit.
    /// `None` when the shape does not fit it (k % 32, k > 12288, unaligned
    /// rows); the caller then takes the two-launch path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_fp8_token(&self, x: &CudaView<'_, f32>, w: &CudaSlice<u8>, s: &CudaSlice<u8>, n: usize, k: usize, round: bool, quant: bool) -> Result<Option<CudaSlice<f32>>> {
        if !k.is_multiple_of(32) || k > 12288 || !self.addr(&w.as_view()).is_multiple_of(16) {
            return Ok(None);
        }
        assert!(x.len() >= k && w.len() >= n * k && s.len() >= n.div_ceil(32) * (k / 32));
        let mut y = self.alloc::<f32>(n)?;
        // enough rows per block that each block's quantization of x stays
        // small next to its rows, while leaving a few blocks per SM; a
        // multiple of 16 (8 warps, two rows each)
        let rows_per_block = (n.div_ceil(340).div_ceil(16) * 16).clamp(16, 64);
        let (ni, ki, ri, ro, qu) = (n as i32, k as i32, rows_per_block as i32, i32::from(round), i32::from(quant));
        let mut b = self.stream.launch_builder(self.f("gemv_fp8_token"));
        b.arg(x).arg(w).arg(s).arg(&mut y).arg(&ni).arg(&ki).arg(&ri).arg(&ro).arg(&qu);
        let cfg = LaunchConfig { grid_dim: (n.div_ceil(rows_per_block) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (k * 4) as u32 };
        // SAFETY: gemv_fp8_token(const float* x, const u8* w, const u8* s, float* y, int n, int k, int rows_per_block,
        // int round, int quant); lengths asserted above, rows 16-byte aligned (base checked, k % 32 == 0),
        // k floats of shared memory (<= 48 KB).
        cu(unsafe { b.launch(cfg) })?;
        Ok(Some(y))
    }

    /// [`gemv_bf16`](Self::gemv_bf16) on fp8 weights `[n][k]` with 32x32
    /// e8m0 tile scales `s`, dequantized on the fly: the same results as on
    /// their bf16 expansion, from half the bytes. `x` is not quantized.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_fp8w(
        &self,
        x: &CudaView<'_, f32>,
        w: &CudaSlice<u8>,
        s: &CudaSlice<u8>,
        y: &mut CudaViewMut<'_, f32>,
        n: usize,
        k: usize,
        nt: usize,
        round: bool,
        group_rows: usize,
        x_stride: usize,
    ) -> Result<()> {
        let xs = if group_rows > 0 { x_stride } else { k };
        let groups = if group_rows > 0 { n.div_ceil(group_rows) } else { 1 };
        assert!(w.len() >= n * k && s.len() >= n.div_ceil(32) * k.div_ceil(32) && y.len() >= nt * n);
        // grouped: row r reads x[t * x_stride + (r / group_rows) * k ..][..k]
        let need = if group_rows > 0 { (nt - 1) * xs + groups * k } else { nt * k };
        assert!(x.len() >= need && (group_rows == 0 || groups * k <= xs));
        let (ni, ki, ti, ro, gi, xi) = (n as i32, k as i32, nt as i32, i32::from(round), group_rows as i32, x_stride as i32);
        let mut b = self.stream.launch_builder(self.f("gemv_fp8w"));
        b.arg(x).arg(w).arg(s).arg(y).arg(&ni).arg(&ki).arg(&ti).arg(&ro).arg(&gi).arg(&xi);
        // SAFETY: gemv_fp8w(const float* x, const u8* w, const u8* s, float* y, int n, int k, int nt, int round,
        // int group_rows, int x_stride); buffer lengths asserted above; one warp per output row.
        cu(unsafe { b.launch(rows_cfg(n)) })?;
        Ok(())
    }

    /// bf16 weight `[n, k]`; `group_rows > 0` = block-diagonal (x row `x_stride` wide).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16(
        &self,
        x: &CudaView<'_, f32>,
        w: &CudaSlice<u16>,
        y: &mut CudaViewMut<'_, f32>,
        n: usize,
        k: usize,
        nt: usize,
        round: bool,
        group_rows: usize,
        x_stride: usize,
    ) -> Result<()> {
        assert!(w.len() >= n * k && y.len() >= n * nt);
        let (ni, ki, ti, ri, gi, si) = (n as i32, k as i32, nt as i32, i32::from(round), group_rows as i32, x_stride as i32);
        let mut b = self.stream.launch_builder(self.f("gemv_bf16"));
        b.arg(x).arg(w).arg(y).arg(&ni).arg(&ki).arg(&ti).arg(&ri).arg(&gi).arg(&si);
        // SAFETY: gemv_bf16(const float* x, const u16* w, float* y, int n, int k, int nt, int round,
        // int group_rows, int x_stride); w/y checked above.
        cu(unsafe { b.launch(rows_cfg(n)) })?;
        Ok(())
    }

    pub fn gemv_f32(&self, x: &CudaView<'_, f32>, w: &CudaSlice<f32>, y: &mut CudaViewMut<'_, f32>, n: usize, k: usize, nt: usize) -> Result<()> {
        assert!(w.len() >= n * k && y.len() >= n * nt);
        let (ni, ki, ti) = (n as i32, k as i32, nt as i32);
        let mut b = self.stream.launch_builder(self.f("gemv_f32"));
        b.arg(x).arg(w).arg(y).arg(&ni).arg(&ki).arg(&ti);
        // SAFETY: gemv_f32(const float* x, const float* w, float* y, int n, int k, int nt).
        cu(unsafe { b.launch(rows_cfg(n)) })?;
        Ok(())
    }

    /// Packed e2m1 `[n, k/2]` with per-32 e8m0 scales `[n, k/32]`, as byte views
    /// into an expert record.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_fp4(
        &self,
        xq: &CudaView<'_, f32>,
        w: &CudaView<'_, u8>,
        s: &CudaView<'_, u8>,
        y: &mut CudaViewMut<'_, f32>,
        n: usize,
        k: usize,
        nt: usize,
        round: bool,
    ) -> Result<()> {
        assert!(k.is_multiple_of(32) && w.len() >= n * k / 2 && s.len() >= n * k / 32 && y.len() >= n * nt);
        let (ni, ki, ti, ri) = (n as i32, k as i32, nt as i32, i32::from(round));
        let mut b = self.stream.launch_builder(self.f("gemv_fp4"));
        b.arg(xq).arg(w).arg(s).arg(y).arg(&ni).arg(&ki).arg(&ti).arg(&ri);
        // SAFETY: gemv_fp4(const float* x, const u8* w, const u8* s, float* y, int n, int k, int nt, int round).
        cu(unsafe { b.launch(rows_cfg(n)) })?;
        Ok(())
    }

    // ------------------------------------------------------------ elementwise

    /// h = bf16(silu(min(g, lim)) * clamp(u, lim) * w[t]); `w` one weight per token.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        w: Option<&CudaSlice<f32>>,
        h: &mut CudaSlice<f32>,
        inter: usize,
        nt: usize,
        lim: f32,
    ) -> Result<()> {
        let n = inter * nt;
        assert!(gate.len() >= n && up.len() >= n && h.len() >= n && w.is_none_or(|w| w.len() >= nt));
        let (ii, ti) = (inter as i32, nt as i32);
        let null = 0u64;
        let mut b = self.stream.launch_builder(self.f("swiglu"));
        b.arg(gate).arg(up);
        match w {
            Some(w) => b.arg(w),
            None => b.arg(&null),
        };
        b.arg(h).arg(&ii).arg(&ti).arg(&lim);
        // SAFETY: swiglu(const float* g, const float* u, const float* w (nullable), float* h, int inter,
        // int nt, float lim); a null w is a 0 pointer-sized argument, which the kernel tests.
        cu(unsafe { b.launch(elems_cfg(n)) })?;
        Ok(())
    }

    /// Raw device address of a view (for pointer tables handed to kernels).
    pub fn addr<T>(&self, v: &CudaView<'_, T>) -> u64 {
        let (p, _sync) = cudarc::driver::DevicePtr::device_ptr(v, &self.stream);
        p
    }

    /// Grouped decode stage 1: every selected expert's gate/up rows, SwiGLU,
    /// clamps and route weight, for one token. `tab` is the `3 * nexp` table
    /// of [`moe_table`](Self::moe_table); each address a live VRAM cache slot.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gate_up(&self, xq: &CudaView<'_, f32>, tab: &CudaSlice<u64>, h: &mut CudaSlice<f32>, nexp: usize, inter: usize, dim: usize, lim: f32) -> Result<()> {
        assert!(xq.len() >= dim && tab.len() >= 3 * nexp && h.len() >= nexp * inter);
        assert!(dim.is_multiple_of(32) && inter.is_multiple_of(32));
        let (ni, ii, di) = (nexp as i32, inter as i32, dim as i32);
        let mut b = self.stream.launch_builder(self.f("moe_gate_up"));
        b.arg(xq).arg(tab).arg(h).arg(&ni).arg(&ii).arg(&di).arg(&lim);
        // SAFETY: moe_gate_up(const float* xq, const u64* tab, float* h, int nexp, int inter, int dim, float lim);
        // every address in tab points at a whole record in a live allocation on this device, pinned
        // for the batch, and only this stream writes those slots, in order.
        cu(unsafe { b.launch(rows_cfg(nexp * inter)) })?;
        Ok(())
    }

    /// Grouped decode stage 2: row `r` of every expert's w2 against its
    /// quantized `hq` row, written to output row `tab[nexp + s]` of `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(&self, hq: &CudaSlice<f32>, tab: &CudaSlice<u64>, out: &mut CudaSlice<f32>, nexp: usize, rows: usize, inter: usize, dim: usize) -> Result<()> {
        assert!(hq.len() >= nexp * inter && tab.len() >= 3 * nexp && out.len() >= rows * dim);
        let (ni, ii, di) = (nexp as i32, inter as i32, dim as i32);
        let mut b = self.stream.launch_builder(self.f("moe_down"));
        b.arg(hq).arg(tab).arg(out).arg(&ni).arg(&ii).arg(&di);
        // SAFETY: moe_down(const float* hq, const u64* tab, float* out, int nexp, int inter, int dim);
        // record addresses as in moe_gate_up; the caller's table holds output rows < `rows`.
        // A half-warp per row: 16 rows per 256-thread block, whole warps.
        cu(unsafe { b.launch(rows_cfg((nexp * dim).div_ceil(2))) })?;
        Ok(())
    }

    /// The host table for [`moe_gate_up`](Self::moe_gate_up) / [`moe_down`](Self::moe_down):
    /// record addresses, then output rows, then route-weight bits.
    pub fn moe_table(recs: &[u64], rows: &[usize], weights: &[f32]) -> Vec<u64> {
        assert!(recs.len() == rows.len() && rows.len() == weights.len());
        recs.iter().copied().chain(rows.iter().map(|&r| r as u64)).chain(weights.iter().map(|w| w.to_bits() as u64)).collect()
    }

    /// y = bf16(sum over slots of out + shared).
    pub fn moe_reduce(&self, out: &CudaSlice<f32>, shared: &CudaSlice<f32>, y: &mut CudaSlice<f32>, nexp: usize, dim: usize) -> Result<()> {
        assert!(out.len() >= nexp * dim && shared.len() >= dim && y.len() >= dim);
        let (ni, di) = (nexp as i32, dim as i32);
        let mut b = self.stream.launch_builder(self.f("moe_reduce"));
        b.arg(out).arg(shared).arg(y).arg(&ni).arg(&di);
        // SAFETY: moe_reduce(const float* out, const float* shared, float* y, int nexp, int dim).
        cu(unsafe { b.launch(elems_cfg(dim)) })?;
        Ok(())
    }

    /// [`moe_reduce`](Self::moe_reduce) with the rows in `host_mask` read from
    /// a [`Handoff`](crate::handoff::Handoff), after waiting on the device for
    /// its flag to reach `seq`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_reduce_host(
        &self,
        out: &CudaSlice<f32>,
        shared: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        nexp: usize,
        dim: usize,
        handoff: &crate::handoff::Handoff,
        host_mask: u32,
        seq: u32,
    ) -> Result<()> {
        assert!(out.len() >= nexp * dim && shared.len() >= dim && y.len() >= dim);
        assert!(nexp <= handoff.rows() && nexp <= 32);
        let (ni, di, rows, flag) = (nexp as i32, dim as i32, handoff.rows_dev(), handoff.flag_dev());
        let mut b = self.stream.launch_builder(self.f("moe_reduce_host"));
        b.arg(out).arg(shared).arg(y).arg(&ni).arg(&di).arg(&rows).arg(&host_mask).arg(&flag).arg(&seq);
        // SAFETY: moe_reduce_host(const float* out, const float* shared, float* y, int nexp, int dim,
        // const float* host_rows, unsigned host_mask, const unsigned* flag, unsigned seq); `rows` and
        // `flag` are device addresses of the hand-off's pinned mapping (nexp <= its rows), which outlives
        // the launch: the model keeps it, and the CPU job that releases `seq` holds it too.
        cu(unsafe { b.launch(elems_cfg(dim)) })?;
        Ok(())
    }

    /// Copy `a` then `b` into `inbox` and publish `seq` there, after this
    /// stream's earlier work: the host waits with [`Inbox::wait`](crate::handoff::Inbox::wait).
    pub fn publish(&self, a: &CudaView<'_, f32>, b: &CudaView<'_, f32>, inbox: &crate::handoff::Inbox, seq: u32) -> Result<()> {
        assert!(a.len() + b.len() <= inbox.floats());
        let (na, nb, dst, flag) = (a.len() as i32, b.len() as i32, inbox.data_dev(), inbox.flag_dev());
        let mut l = self.stream.launch_builder(self.f("publish"));
        l.arg(a).arg(&na).arg(b).arg(&nb).arg(&dst).arg(&flag).arg(&seq);
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: publish(const float* a, int na, const float* b, int nb, float* dst, unsigned* flag,
        // unsigned seq); dst and flag are device addresses of the inbox's pinned mapping, which holds
        // na + nb floats (asserted) and outlives the launch (the model keeps it).
        cu(unsafe { l.launch(cfg) })?;
        Ok(())
    }

    /// dst[t] = src[idx[t]] for `nt` rows of `d`.
    pub fn gather_rows(&self, src: &CudaView<'_, f32>, idx: &CudaSlice<i32>, dst: &mut CudaSlice<f32>, d: usize, nt: usize) -> Result<()> {
        assert!(dst.len() >= d * nt && idx.len() >= nt);
        let (di, ti) = (d as i32, nt as i32);
        let mut b = self.stream.launch_builder(self.f("gather_rows"));
        b.arg(src).arg(idx).arg(dst).arg(&di).arg(&ti);
        // SAFETY: gather_rows(const float* src, const int* idx, float* dst, int d, int nt); every idx[t]
        // is a token row of src the caller built from its own routes.
        cu(unsafe { b.launch(elems_cfg(d * nt)) })?;
        Ok(())
    }

    /// A fresh device copy of `src`.
    pub fn dup(&self, src: &CudaView<'_, f32>) -> Result<CudaSlice<f32>> {
        cu(self.stream.clone_dtod(src))
    }

    /// Device-to-device copy of `src` into the front of `dst`.
    pub fn copy(&self, src: &CudaView<'_, f32>, dst: &mut CudaViewMut<'_, f32>) -> Result<()> {
        cu(self.stream.memcpy_dtod(src, dst))
    }

    /// acc[idx[t]] += src[t] for `nt` rows of `d`.
    pub fn scatter_add_rows(&self, acc: &mut CudaSlice<f32>, src: &CudaSlice<f32>, idx: &CudaSlice<i32>, d: usize, nt: usize) -> Result<()> {
        assert!(src.len() >= d * nt && idx.len() >= nt);
        let (di, ti) = (d as i32, nt as i32);
        let mut b = self.stream.launch_builder(self.f("scatter_add_rows"));
        b.arg(acc).arg(src).arg(idx).arg(&di).arg(&ti);
        // SAFETY: scatter_add_rows(float* acc, const float* src, const int* idx, int d, int nt);
        // every idx[t] is a token row the caller built from its own routes (< acc rows).
        cu(unsafe { b.launch(elems_cfg(d * nt)) })?;
        Ok(())
    }

    /// y = bf16(a + b).
    pub fn add_round(&self, a: &CudaSlice<f32>, b_: &CudaView<'_, f32>, y: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        assert!(a.len() >= n && y.len() >= n);
        let nl = n as i64;
        let mut b = self.stream.launch_builder(self.f("add_round"));
        b.arg(a).arg(b_).arg(y).arg(&nl);
        // SAFETY: add_round(const float* a, const float* b, float* y, long n).
        cu(unsafe { b.launch(elems_cfg(n)) })?;
        Ok(())
    }

    // ------------------------------------------------------------ norms, rope

    /// ops::rmsnorm over `rows` rows of `d`.
    pub fn rmsnorm(&self, x: &CudaView<'_, f32>, w: &CudaSlice<f32>, y: &mut CudaViewMut<'_, f32>, rows: usize, d: usize, eps: f32) -> Result<()> {
        assert!(w.len() >= d && y.len() >= rows * d);
        let di = d as i32;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(self.f("rmsnorm"));
        b.arg(x).arg(w).arg(y).arg(&di).arg(&eps);
        // SAFETY: rmsnorm(const float* x, const float* w, float* y, int d, float eps); block = one row,
        // 256 threads with 256 floats of shared memory (power of two for the tree reduction).
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    /// Rotate the `half` pairs at `x[r*stride + off ..]` to `pos[r]` for every row.
    #[allow(clippy::too_many_arguments)]
    /// Rotate the last `2 * half` values of `rows` rows (every `stride`
    /// floats, from `off`). Positions: see [`Pos`].
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        x: &mut CudaViewMut<'_, f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        pos: Pos<'_>,
        rows: usize,
        stride: usize,
        off: usize,
        half: usize,
        inverse: bool,
    ) -> Result<()> {
        assert!(x.len() >= (rows - 1) * stride + off + 2 * half);
        let (ri, si, oi, hi, ii) = (rows as i32, stride as i32, off as i32, half as i32, i32::from(inverse));
        let null: u64 = 0;
        let mut b = self.stream.launch_builder(self.f("rope"));
        b.arg(x).arg(cos).arg(sin);
        let (p0, per) = match pos {
            Pos::Rows(p) => {
                assert!(p.len() >= rows);
                b.arg(p);
                (0i32, 1i32)
            }
            Pos::Linear { base, per } => {
                assert!(per > 0);
                b.arg(&null);
                (base as i32, per as i32)
            }
        };
        b.arg(&ri).arg(&si).arg(&oi).arg(&hi).arg(&ii).arg(&p0).arg(&per);
        // SAFETY: rope(float* x, const float* cos, const float* sin, const int* pos (or null), int rows,
        // int stride, int off, int half, int inverse, int pos0, int per); x covers every row touched
        // (asserted), every position is inside the tables (the model sizes them for max_seq).
        cu(unsafe { b.launch(elems_cfg(rows * half)) })?;
        Ok(())
    }

    // ------------------------------------------------------------ hyper-connections

    /// Per token: the 24 projections of its `n`-wide stream, then its sum of squares.
    pub fn hc_project(&self, x: &CudaSlice<f32>, fn_: &CudaSlice<f32>, out: &mut CudaSlice<f32>, n: usize, nt: usize) -> Result<()> {
        assert!(x.len() >= n * nt && fn_.len() >= 24 * n && out.len() >= 25 * nt);
        let ni = n as i32;
        let cfg = LaunchConfig { grid_dim: (nt as u32, 25, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
        let mut b = self.stream.launch_builder(self.f("hc_project"));
        b.arg(x).arg(fn_).arg(out).arg(&ni);
        // SAFETY: hc_project(const float* x, const float* fn, float* out, int n); block per (token, output).
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    /// [`hc_project`](Self::hc_project) then [`hc_mix`](Self::hc_mix) in one
    /// launch (same results). `counter` holds `nt` zeros and is left so.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_project_mix(
        &self,
        x: &CudaSlice<f32>,
        fn_: &CudaSlice<f32>,
        proj: &mut CudaSlice<f32>,
        base: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        mix: &mut CudaSlice<f32>,
        counter: &mut CudaSlice<u32>,
        nt: usize,
        n: usize,
        norm_eps: f32,
        iters: usize,
        hc_eps: f32,
    ) -> Result<()> {
        assert!(x.len() >= n * nt && fn_.len() >= 24 * n && proj.len() >= 25 * nt && mix.len() >= 24 * nt && counter.len() >= nt);
        assert!(base.len() >= 24 && scale.len() >= 3);
        let (ni, ii) = (n as i32, iters as i32);
        let cfg = LaunchConfig { grid_dim: (nt as u32, 25, 1), block_dim: (256, 1, 1), shared_mem_bytes: 256 * 4 };
        let mut b = self.stream.launch_builder(self.f("hc_project_mix"));
        b.arg(x).arg(fn_).arg(proj).arg(&ni).arg(base).arg(scale).arg(mix).arg(&norm_eps).arg(&ii).arg(&hc_eps).arg(counter);
        // SAFETY: hc_project_mix(const float* x, const float* fn, float* out, int n, const float* base,
        // const float* scale, float* mix, float norm_eps, int iters, float hc_eps, unsigned* counter);
        // lengths asserted; a block per (token, output), counter[t] zero on entry (the last block resets it).
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    /// hc::mixes_from_projection on the device: `proj` `[nt][25]` (from
    /// [`hc_project`](Self::hc_project)) to `mix` `[nt][24]` = pre, post, comb.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix(
        &self,
        proj: &CudaSlice<f32>,
        base: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        mix: &mut CudaSlice<f32>,
        nt: usize,
        n: usize,
        norm_eps: f32,
        iters: usize,
        hc_eps: f32,
    ) -> Result<()> {
        assert!(proj.len() >= 25 * nt && base.len() >= 24 && scale.len() >= 3 && mix.len() >= 24 * nt);
        let (ti, ni, ii) = (nt as i32, n as i32, iters as i32);
        let mut b = self.stream.launch_builder(self.f("hc_mix"));
        b.arg(proj).arg(base).arg(scale).arg(mix).arg(&ti).arg(&ni).arg(&norm_eps).arg(&ii).arg(&hc_eps);
        // SAFETY: hc_mix(const float* proj, const float* base, const float* scale, float* mix, int nt,
        // int n, float norm_eps, int iters, float hc_eps); 16 lanes per token, whole warps
        // (elems_cfg rounds up to 256-thread blocks), out-of-range tokens only shuffle.
        cu(unsafe { b.launch(elems_cfg(nt * 16)) })?;
        Ok(())
    }

    /// [`hc_pre`](Self::hc_pre) then [`rmsnorm`](Self::rmsnorm) with weight
    /// `w`, fused (same results): `y[t] = rmsnorm(sum_i pre[t][i] * x[t][i])`.
    /// Both use the same 1024-thread reduction.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_norm(&self, x: &CudaSlice<f32>, mix: &CudaView<'_, f32>, stride: usize, w: &CudaSlice<f32>, y: &mut CudaSlice<f32>, d: usize, nt: usize, eps: f32) -> Result<()> {
        assert!(x.len() >= 4 * d * nt && mix.len() >= (nt - 1) * stride + 4 && w.len() >= d && y.len() >= d * nt);
        assert!(d * 4 + 128 <= 48 * 1024);
        let (si, di) = (stride as i32, d as i32);
        let cfg = LaunchConfig { grid_dim: (nt as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: (d * 4) as u32 };
        let mut b = self.stream.launch_builder(self.f("hc_pre_norm"));
        b.arg(x).arg(mix).arg(&si).arg(w).arg(y).arg(&di).arg(&eps);
        // SAFETY: hc_pre_norm(const float* x, const float* mix, int stride, const float* w, float* y, int d,
        // float eps); lengths asserted; a 1024-thread block per token with d floats of shared memory.
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    /// Decode's kv row into its window slot: rmsnorm with `w`, rope of the last
    /// `2 * half` values at `pos`, fp8 act-quant, write to `dst` (`d` floats);
    /// the same bits as those four steps apart.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_finish(&self, kv0: &CudaView<'_, f32>, w: &CudaSlice<f32>, cos: &CudaSlice<f32>, sin: &CudaSlice<f32>, pos: usize, half: usize, dst: &mut CudaViewMut<'_, f32>, d: usize, eps: f32) -> Result<()> {
        assert!(d <= 1024 && d.is_multiple_of(32) && 2 * half <= d && kv0.len() >= d && w.len() >= d && dst.len() >= d);
        assert!(cos.len() >= (pos + 1) * half && sin.len() >= (pos + 1) * half);
        let (pi, hi, di) = (pos as i32, half as i32, d as i32);
        let threads = d.div_ceil(32) * 32;
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (threads.max(32) as u32, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.stream.launch_builder(self.f("kv_finish"));
        b.arg(kv0).arg(w).arg(cos).arg(sin).arg(&pi).arg(&hi).arg(dst).arg(&di).arg(&eps);
        // SAFETY: kv_finish(const float* kv0, const float* w, const float* cos, const float* sin, int pos, int half,
        // float* dst, int d, float eps); lengths asserted; one block, a thread per value.
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    /// `pre` of token t starts at `mix[t * stride]` (24 for an hc_mix buffer).
    pub fn hc_pre(&self, x: &CudaSlice<f32>, mix: &CudaView<'_, f32>, y: &mut CudaSlice<f32>, d: usize, nt: usize, stride: usize) -> Result<()> {
        assert!(x.len() >= 4 * d * nt && mix.len() >= stride * (nt - 1) + 4 && y.len() >= d * nt);
        let (di, ti, si) = (d as i32, nt as i32, stride as i32);
        let mut b = self.stream.launch_builder(self.f("hc_pre"));
        b.arg(x).arg(mix).arg(y).arg(&di).arg(&ti).arg(&si);
        // SAFETY: hc_pre(const float* x, const float* mix, float* y, int d, int nt, int stride).
        cu(unsafe { b.launch(elems_cfg(d * nt)) })?;
        Ok(())
    }

    /// `mix` per token as hc_mix writes it: pre[4], post[4], comb[16] (row = residual copy).
    pub fn hc_post(&self, out: &CudaSlice<f32>, res: &CudaSlice<f32>, mix: &CudaSlice<f32>, y: &mut CudaSlice<f32>, d: usize, nt: usize) -> Result<()> {
        assert!(out.len() >= d * nt && res.len() >= 4 * d * nt && mix.len() >= 24 * nt && y.len() >= 4 * d * nt);
        let (di, ti) = (d as i32, nt as i32);
        let mut b = self.stream.launch_builder(self.f("hc_post"));
        b.arg(out).arg(res).arg(mix).arg(y).arg(&di).arg(&ti);
        // SAFETY: hc_post(const float* out, const float* res, const float* mix, float* y, int d, int nt).
        cu(unsafe { b.launch(elems_cfg(4 * d * nt)) })?;
        Ok(())
    }

    // ------------------------------------------------------------ attention

    /// Every (token, head) over its index list; `kv` rows are `hd` wide.
    #[allow(clippy::too_many_arguments)]
    /// Sparse attention over kv rows gathered by `idx` (`-1` = none): row
    /// `p` is `kva[p]` below `rows_a`, else `kvb[p - rows_a]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        q: &CudaSlice<f32>,
        kva: &CudaView<'_, f32>,
        rows_a: usize,
        kvb: &CudaView<'_, f32>,
        idx: &CudaSlice<i32>,
        sink: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        nt: usize,
        nh: usize,
        hd: usize,
        nidx: usize,
        scale: f32,
        rope: Option<(&CudaSlice<f32>, &CudaSlice<f32>, usize, usize)>,
    ) -> Result<()> {
        assert!(q.len() >= nt * nh * hd && idx.len() >= nt * nidx && sink.len() >= nh && out.len() >= nt * nh * hd);
        assert!(kva.len() >= rows_a * hd);
        let (hi, di, ii, ra) = (nh as i32, hd as i32, nidx as i32, rows_a as i32);
        // a thread per output column (up to 1024), whole warps; shared memory:
        // nidx scores, 32 per-warp maxima and the denominator
        let threads = hd.div_ceil(32).clamp(1, 32) * 32;
        let cfg = LaunchConfig { grid_dim: ((nt * nh) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: ((nidx.max(1) + 33) * 4) as u32 };
        let mut b = self.stream.launch_builder(self.f("sparse_attn"));
        b.arg(q).arg(kva).arg(&ra).arg(kvb).arg(idx).arg(sink).arg(out).arg(&hi).arg(&di).arg(&ii).arg(&scale);
        let null: u64 = 0;
        let (half, pos0) = match rope {
            Some((cos, sin, half, pos0)) => {
                // a thread per column, so rotation pairs are adjacent lanes
                assert!(threads == hd && hd.is_multiple_of(2) && 2 * half <= hd);
                b.arg(cos).arg(sin);
                (half as i32, pos0 as i32)
            }
            None => {
                b.arg(&null).arg(&null);
                (0, 0)
            }
        };
        let off = hd as i32 - 2 * half;
        b.arg(&off).arg(&half).arg(&pos0);
        // SAFETY: sparse_attn(const float* q, const float* kva, int rows_a, const float* kvb, const int* idx,
        // const float* sink, float* out, int nh, int hd, int nidx, float scale, const float* cos (or null),
        // const float* sin, int rope_off, int rope_half, int pos0); every non-negative idx
        // names a row of kva (< rows_a, asserted to fit) or of kvb (the caller's compressed cache, which
        // holds every row it indexes); shared memory holds nidx scores + 33 floats.
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn index_scores(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaView<'_, f32>,
        w: &CudaSlice<f32>,
        s: &mut CudaSlice<f32>,
        nh: usize,
        ihd: usize,
        nkeys: usize,
        nt: usize,
        wscale: f32,
    ) -> Result<()> {
        assert!(q.len() >= nt * nh * ihd && k.len() >= nkeys * ihd && w.len() >= nt * nh && s.len() >= nt * nkeys);
        if nkeys * nt == 0 {
            return Ok(());
        }
        let (hi, di, ki, ti) = (nh as i32, ihd as i32, nkeys as i32, nt as i32);
        let mut b = self.stream.launch_builder(self.f("index_scores"));
        b.arg(q).arg(k).arg(w).arg(s).arg(&hi).arg(&di).arg(&ki).arg(&ti).arg(&wscale);
        // SAFETY: index_scores(const float* q, const float* k, const float* w, float* s, int nh, int ihd,
        // int nkeys, int nt, float wscale); one block per (token, key), a thread per head, nh floats of shared memory.
        assert!(nh <= 1024);
        let threads = nh.div_ceil(32) * 32;
        let cfg = LaunchConfig { grid_dim: ((nkeys * nt) as u32, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: (nh * 4) as u32 };
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }

    pub fn compress_pool(&self, kv: &CudaView<'_, f32>, score: &CudaView<'_, f32>, out: &mut CudaSlice<f32>, groups: usize, r: usize, hd: usize) -> Result<()> {
        assert!(r <= 8 && out.len() >= groups * hd);
        let (gi, ri, hi) = (groups as i32, r as i32, hd as i32);
        let mut b = self.stream.launch_builder(self.f("compress_pool"));
        b.arg(kv).arg(score).arg(out).arg(&gi).arg(&ri).arg(&hi);
        // SAFETY: compress_pool(const float* kv, const float* score, float* out, int groups, int r, int hd);
        // kv/score hold groups*r rows (caller's slices), r <= 8 fits the kernel's local array.
        cu(unsafe { b.launch(elems_cfg(groups * hd)) })?;
        Ok(())
    }

    // ------------------------------------------------------------ engram

    #[allow(clippy::too_many_arguments)]
    pub fn engram_gate(&self, h: &CudaSlice<f32>, kv: &CudaSlice<f32>, qk: &CudaSlice<f32>, out: &mut CudaSlice<f32>, d: usize, nt: usize, eps: f32, scale: f32) -> Result<()> {
        assert!(h.len() >= 4 * d * nt && kv.len() >= 5 * d * nt && qk.len() >= 4 * d && out.len() >= 4 * d * nt);
        let di = d as i32;
        let cfg = LaunchConfig { grid_dim: ((4 * nt) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 3 * 256 * 4 };
        let mut b = self.stream.launch_builder(self.f("engram_gate"));
        b.arg(h).arg(kv).arg(qk).arg(out).arg(&di).arg(&eps).arg(&scale);
        // SAFETY: engram_gate(const float* h, const float* kv, const float* qk, float* out, int d,
        // float eps, float scale); block per (token, copy), 3*256 floats of shared memory.
        cu(unsafe { b.launch(cfg) })?;
        Ok(())
    }
}
