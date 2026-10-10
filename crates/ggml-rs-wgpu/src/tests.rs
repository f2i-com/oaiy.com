use super::*;

/// Byte offsets of each type's f16 scale fields inside a block.
fn scale_offsets(dtype: GgmlType) -> &'static [usize] {
    match dtype {
        GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => &[0, 2],
        GgmlType::Q2_K => &[80, 82],
        GgmlType::Q3_K => &[108],
        GgmlType::Q6_K => &[208],
        _ => &[0],
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Random blocks with finite, modest f16 scales.
fn random_weights(dtype: GgmlType, rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    // ggml's blocks (the GPU's may be padded)
    let (elems, bytes) = (dtype.block_size() as u32, dtype.type_size() as u32);
    let blocks = rows * k / elems as usize;
    let mut out = vec![0u8; blocks * bytes as usize];
    for b in out.iter_mut() {
        *b = rng.next() as u8;
    }
    for blk in 0..blocks {
        // (IQ1_M has no scale field: its f16 is the top nibbles of its last four u16s)
        if dtype == GgmlType::IQ1_M {
            let bits = half::f16::from_f32(0.002 + 0.02 * rng.unit()).to_bits();
            for i in 0..4 {
                let at = blk * bytes as usize + 48 + 2 * i + 1;
                out[at] = (out[at] & 0x0f) | ((((bits >> (4 * i)) & 15) as u8) << 4);
            }
            continue;
        }
        for &off in scale_offsets(dtype) {
            let scale = half::f16::from_f32(0.002 + 0.02 * rng.unit()).to_bits().to_le_bytes();
            let at = blk * bytes as usize + off;
            out[at..at + 2].copy_from_slice(&scale);
        }
    }
    out
}

fn reference(dtype: GgmlType, w: &[u8], rows: usize, k: usize, x: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
    let mut dense = vec![0f32; rows * k];
    ggml_quants::dequantize(dtype, w, &mut dense).unwrap();
    let mut y = vec![0f32; m * rows];
    let mut mag = vec![0f32; m * rows];
    for mi in 0..m {
        for r in 0..rows {
            let (mut s, mut a) = (0f64, 0f64);
            for i in 0..k {
                let p = dense[r * k + i] as f64 * x[mi * k + i] as f64;
                s += p;
                a += p.abs();
            }
            y[mi * rows + r] = s as f32;
            mag[mi * rows + r] = a as f32;
        }
    }
    (y, mag)
}

fn backend() -> Option<WgpuBackend> {
    match WgpuBackend::new(Some(4 * GIB)) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping WebGPU tests: {e}");
            None
        }
    }
}

// (the grid types, IQ2_XXS to IQ1_M, are off the GPU until their tables are a buffer's: `shaders::layout`)
const TYPES: [GgmlType; 13] = [
    GgmlType::Q2_0, GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0, GgmlType::IQ4_NL,
    GgmlType::Q2_K, GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::IQ4_XS,
];

#[test]
fn every_quant_type_matches_the_cpu_dequantization() {
    let Some(b) = backend() else { return };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (rows, k) = (70, 512);
    for dtype in TYPES {
        let w = random_weights(dtype, rows, k, &mut rng);
        let qt = b.to_device_quant(QuantizedTensor::from_bytes_cpu(w.clone(), vec![rows, k], dtype));
        assert!(qt.is_device(), "{dtype:?} was not uploaded");
        // 1 and 5 tokens take the one-row kernel, 9 and 70 the tiled one (70: a second tile of tokens, part empty).
        for m in [1, 5, 9, 70] {
            let x: Vec<f32> = (0..m * k).map(|_| rng.unit() * 2.0 - 1.0).collect();
            let y = b.linear_q(&Tensor::from_vec(x.clone(), vec![m, k]), &qt);
            assert_eq!(y.shape(), &[m, rows]);
            let (want, mag) = reference(dtype, &w, rows, k, &x, m);
            for i in 0..m * rows {
                let err = (y.data()[i] - want[i]).abs();
                assert!(err <= 1e-4 * mag[i] + 1e-5, "{dtype:?} m={m} out {i}: gpu {} vs cpu {} (|terms| {})", y.data()[i], want[i], mag[i]);
            }
        }
        // Read-back returns the uploaded bytes unchanged.
        assert_eq!(qt.to_host().bytes(), &w[..], "{dtype:?} round trip");
    }
}

/// What a quantized projection costs: a 4096 x 4096 weight against 1 to 512 tokens, each call a submit and a read
/// back (its time with the round trip in it), for the common GGUF types.
#[test]
#[ignore = "a timing; run with --nocapture"]
fn measure_quantized_projections() {
    let Some(b) = backend() else { return };
    let mut rng = Rng(11);
    let (rows, k) = (4096, 4096);
    for dtype in [GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::Q8_0] {
        let w = random_weights(dtype, rows, k, &mut rng);
        let bytes = w.len();
        let qt = b.to_device_quant(QuantizedTensor::from_bytes_cpu(w, vec![rows, k], dtype));
        for m in [1usize, 8, 64, 512] {
            let x = Tensor::from_vec((0..m * k).map(|_| rng.unit() * 2.0 - 1.0).collect(), vec![m, k]);
            b.linear_q(&x, &qt);
            let calls = if m >= 64 { 5 } else { 20 };
            let t = std::time::Instant::now();
            for _ in 0..calls {
                b.linear_q(&x, &qt);
            }
            let secs = t.elapsed().as_secs_f64() / calls as f64;
            eprintln!(
                "{dtype:?} [{rows}, {k}] ({:.1} MB) x {m} tokens: {:.3} ms a call, {:.0} GFLOP/s, {:.0} GB/s of weights",
                bytes as f64 / 1e6,
                secs * 1e3,
                2.0 * (m * rows * k) as f64 / secs / 1e9,
                bytes as f64 / secs / 1e9
            );
        }
    }
}

#[test]
fn weights_beyond_the_budget_stay_on_the_host_and_still_compute() {
    let Ok(b) = WgpuBackend::new(Some(0)) else { return };
    let mut rng = Rng(7);
    let (rows, k) = (8, 256);
    let w = random_weights(GgmlType::Q4_K, rows, k, &mut rng);
    let qt = b.try_to_device_quant(QuantizedTensor::from_bytes_cpu(w.clone(), vec![rows, k], GgmlType::Q4_K), 0);
    assert!(!qt.is_device());
    assert_eq!(b.usage().0, 0);
    let x: Vec<f32> = (0..k).map(|_| rng.unit()).collect();
    let y = b.linear_q(&Tensor::from_vec(x.clone(), vec![1, k]), &qt);
    let (want, mag) = reference(GgmlType::Q4_K, &w, rows, k, &x, 1);
    for i in 0..rows {
        assert!((y.data()[i] - want[i]).abs() <= 1e-4 * mag[i] + 1e-5);
    }
}

#[test]
fn usage_is_returned_when_weights_are_dropped() {
    let Some(b) = backend() else { return };
    let w = random_weights(GgmlType::Q8_0, 4, 64, &mut Rng(3));
    let qt = b.to_device_quant(QuantizedTensor::from_bytes_cpu(w, vec![4, 64], GgmlType::Q8_0));
    assert_eq!(b.usage().0, 4 * 2 * 34);
    drop(qt);
    assert_eq!(b.usage().0, 0);
}

#[test]
fn a_discrete_cards_default_budget_leaves_room() {
    use super::discrete_budget;
    const G: u64 = 1 << 30;
    assert_eq!(discrete_budget(32 * G), 28 * G);
    assert_eq!(discrete_budget(8 * G), 4 * G);
    assert_eq!(discrete_budget(6 * G), 3 * G);
    assert_eq!(discrete_budget(0), 0);
}

/// The Metal text of a kernel declares a temporary for each operand of a packed dot product, named after the operand:
/// a kernel that uses one operand in several of them was refused by Apple's compiler ("redefinition of ..."). On
/// Metal each goes through a function of its own. This holds what that rewriting must keep: the kernel is still one
/// naga takes, no dot product is left outside its function, and a kernel with none is not touched.
#[test]
fn a_kernels_packed_dot_products_go_through_a_function_of_their_own_for_metal() {
    use wgpu::naga;
    let valid = |text: &str| {
        let module = naga::front::wgsl::parse_str(text).unwrap_or_else(|e| panic!("WGSL: {}", e.emit_to_string(text)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("validation: {e:?}"));
        module
    };
    let calls = |text: &str, name: &str| text.matches(&format!("{name}(")).count();
    let mut seen = 0;
    for dtype in [GgmlType::Q4_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q8_0] {
        for mr in [1u32, 2, 4] {
            let Some(kernel) = crate::shaders::rb_kernel_q8(dtype, crate::shaders::rb_rows(dtype, mr), mr) else { continue };
            let before = calls(&kernel, "dot4I8Packed");
            assert!(before > 0, "{dtype:?} for {mr} tokens has no packed dot product: the test looks at the wrong kernel");
            let wrapped = packed_dots_wrapped(kernel.clone());
            assert_eq!(calls(&wrapped, "dot4I8Packed"), 1, "{dtype:?}, {mr}: one use is left, the function's own");
            assert_eq!(calls(&wrapped, "oaiy_dot4_i8"), before + 1, "{dtype:?}, {mr}: every use and the function's heading");
            let (was, is) = (valid(&kernel), valid(&wrapped));
            assert_eq!(is.functions.len(), was.functions.len() + 1, "{dtype:?}, {mr}: one function more");
            assert_eq!(workgroup_bytes(&wrapped), workgroup_bytes(&kernel), "{dtype:?}, {mr}: the workgroup's memory is as it was");
            seen += 1;
        }
    }
    assert!(seen >= 6, "only {seen} multi-token kernels were made");
    // The unsigned one, in a kernel of the same shape as the EXL3 ones' use of it.
    let unsigned = "@compute @workgroup_size(1) fn main() { let hx = 0x01020304u; let a = dot4U8Packed(hx, 0x01010101u); let b = dot4U8Packed(hx, 0x02020202u); _ = a + b; }".to_string();
    let wrapped = packed_dots_wrapped(unsigned);
    assert_eq!((calls(&wrapped, "dot4U8Packed"), calls(&wrapped, "oaiy_dot4_u8")), (1, 3));
    valid(&wrapped);
    // A kernel with neither is the text it was.
    let plain = "@compute @workgroup_size(1) fn main() { }".to_string();
    assert_eq!(packed_dots_wrapped(plain.clone()), plain);
}