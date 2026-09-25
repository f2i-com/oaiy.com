//! VENDORED-LOCAL: resident EXL3 projections, original packed bytes.
use crate::CudaBackend;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use ggml_rs::{
    exl3::{Exl3Data, PackedLinear},
    Backend, Tensor,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct Exl3Matrix {
    backend: Arc<CudaBackend>,
    words: CudaSlice<u32>,
    suh: CudaSlice<f32>,
    svh: CudaSlice<f32>,
    input: CudaSlice<u32>,
    output: CudaSlice<u32>,
    shape: [usize; 2],
    tile_words: i32,
}
impl Exl3Matrix {
    pub fn upload(backend: Arc<CudaBackend>, data: Exl3Data) -> Result<Self, String> {
        data.validate()?;
        let s = &backend.stream;
        let mut inverse = vec![0u32; data.output_map.len()];
        for (i, &v) in data.output_map.iter().enumerate() {
            inverse[v as usize] = i as u32;
        }
        let words = s.clone_htod(&data.words).map_err(|e| format!("{e:?}"))?;
        let suh = s.clone_htod(&data.suh).map_err(|e| format!("{e:?}"))?;
        let svh = s.clone_htod(&data.svh).map_err(|e| format!("{e:?}"))?;
        let input = s
            .clone_htod(&data.input_map)
            .map_err(|e| format!("{e:?}"))?;
        let output = s.clone_htod(&inverse).map_err(|e| format!("{e:?}"))?;
        Ok(Self {
            backend,
            words,
            suh,
            svh,
            input,
            output,
            shape: [data.svh.len(), data.suh.len()],
            tile_words: data.tile_words as i32,
        })
    }
}
impl PackedLinear for Exl3Matrix {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.words.len() * 4 + (self.suh.len() + self.svh.len()) * 8
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        self.linear_impl(x, false, None)
    }
}
impl Exl3Matrix {
    /// Scalar-layout CUDA reference for numerical and performance comparisons.
    pub fn linear_reference(&self, x: &Tensor) -> Tensor {
        self.linear_impl(x, true, None)
    }
    /// Explicit split count for CUDA tuning; automatic selection is used by `linear`.
    pub fn linear_with_splits(&self, x: &Tensor, splits: usize) -> Tensor {
        assert!(
            (1..=32).contains(&splits),
            "EXL3 splits must be between 1 and 32"
        );
        self.linear_impl(x, false, Some(splits))
    }
    fn linear_impl(&self, x: &Tensor, reference: bool, requested_splits: Option<usize>) -> Tensor {
        let b = &self.backend;
        let s = &b.stream;
        let (n, k) = (self.shape[0], self.shape[1]);
        assert_eq!(x.shape().last(), Some(&k));
        let m = x.numel() / k;
        let a = b.cuda_input(x);
        // SAFETY: exl3_had writes all m*k elements before any read on this stream.
        let mut xh = unsafe { s.alloc::<f32>(m * k) }.expect("EXL3 input scratch");
        // More independent blocks keep narrow projections busy. The fused output
        // transform reduces partial sums, so splitting adds no kernel launch.
        let tiled = m == 1 && !reference && matches!(self.tile_words, 32 | 48 | 56 | 64 | 96);
        let splits = if tiled {
            requested_splits.unwrap_or_else(|| {
                (4096usize.div_ceil(n / 16))
                    .next_power_of_two()
                    .min(8)
                    .min((k / 256).max(1))
            })
        } else {
            1
        };
        let (ki, ni) = (k as i32, n as i32);
        // SAFETY: validated dimensions/maps bound every access; all buffers
        // live through launches on this backend's ordered compute stream.
        unsafe {
            s.launch_builder(b.func("exl3_had"))
                .arg(a.as_ref())
                .arg(&mut xh)
                .arg(&self.suh)
                .arg(&self.input)
                .arg(&ki)
                .arg(&0i32)
                .launch(LaunchConfig {
                    grid_dim: ((k / 128) as u32, m as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .expect("EXL3 input transform");
        }
        let yh = if m == 1 {
            // SAFETY: the chosen GEMV writes every output/partial before its transform reads it.
            let mut y = unsafe { s.alloc::<f32>(n * splits) }.expect("EXL3 output scratch");
            let kernel = match (reference, self.tile_words) {
                (false, 32) => "exl3_tile_32",
                (false, 48) => "exl3_tile_48",
                (false, 56) => "exl3_tile_56",
                (false, 64) => "exl3_tile_64",
                (false, 96) => "exl3_tile_96",
                (true, 32) => "exl3_gemv_32",
                (true, 48) => "exl3_gemv_48",
                (true, 56) => "exl3_gemv_56",
                (true, 64) => "exl3_gemv_64",
                (true, 96) => "exl3_gemv_96",
                _ => "exl3_gemv_generic",
            };
            let generic = kernel == "exl3_gemv_generic";
            let tiled = !reference && !generic;
            // SAFETY: validated dimensions bound packed reads and each block's
            // output. The fallback takes the validated runtime tile width.
            unsafe {
                let mut launch = s.launch_builder(b.func(kernel));
                launch
                    .arg(&self.words)
                    .arg(&xh)
                    .arg(&mut y)
                    .arg(&ki)
                    .arg(&ni);
                if generic {
                    launch.arg(&self.tile_words);
                }
                launch
                    .launch(LaunchConfig {
                        grid_dim: (
                            (n as u32).div_ceil(if tiled { 16 } else { 8 }),
                            splits as u32,
                            1,
                        ),
                        block_dim: (32, if tiled { 4 } else { 8 }, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("EXL3 GEMV");
            }
            b.make_tensor(y, vec![splits, n])
        } else {
            // SAFETY: exl3_reconstruct writes every n*k element before GEMM reads it.
            let mut w = unsafe { s.alloc::<f32>(n * k) }.expect("EXL3 reconstruction scratch");
            // SAFETY: kernel bounds p against n*k, matching this allocation.
            unsafe {
                s.launch_builder(b.func("exl3_reconstruct"))
                    .arg(&self.words)
                    .arg(&mut w)
                    .arg(&ki)
                    .arg(&ni)
                    .arg(&self.tile_words)
                    .launch(LaunchConfig::for_num_elems((n * k) as u32))
                    .expect("EXL3 reconstruction");
            }
            b.linear(
                &b.make_tensor(xh, vec![m, k]),
                &b.make_tensor(w, vec![n, k]),
            )
        };
        let input = b.cuda_input(&yh);
        // SAFETY: either output transform writes all m*n elements via a validated permutation.
        let mut y = unsafe { s.alloc::<f32>(m * n) }.expect("EXL3 result");
        // SAFETY: same validated block transform, with inverse output permutation.
        unsafe {
            s.launch_builder(b.func(if splits > 1 {
                "exl3_had_reduce"
            } else {
                "exl3_had"
            }))
            .arg(input.as_ref())
            .arg(&mut y)
            .arg(&self.svh)
            .arg(&self.output)
            .arg(&ni)
            .arg(&(if splits > 1 { splits as i32 } else { 1i32 }))
            .launch(LaunchConfig {
                grid_dim: ((n / 128) as u32, m as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
            .expect("EXL3 output transform");
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = n;
        b.make_tensor(y, shape)
    }
}
