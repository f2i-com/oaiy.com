//! Qwen's axial vision RoPE and interleaved temporal/height/width text RoPE.
//! The host implementation is also the numerical reference for GPU kernels.
use ggml_rs::{Backend, Tensor};

pub fn rotate(b: &dyn Backend, x: &mut Tensor, positions: &[[u32; 3]], rotated: usize, theta: f32, axes: &[usize], frequencies: &[usize], frequency_dim: usize) {
    // VENDORED-LOCAL: dispatch to the resident GPU path, with a CPU reference fallback.
    b.rope_axes(x, positions, rotated, theta, axes, frequencies, frequency_dim);
}

pub fn text(b: &dyn Backend, x: &mut Tensor, positions: &[[u32; 3]], rotated: usize, theta: f32) {
    // Qwen3.5/3.6/3.8 dense uses mrope_section=[11,11,10].
    assert_eq!(rotated, 64, "unsupported Qwen MRoPE dimension");
    // VENDORED-LOCAL: generated text after an image has equal spatial axes.
    // Use the resident GPU kernel with the multimodal offset, avoiding two
    // device/host transfers per attention layer on every generated token.
    if positions.iter().all(|p|p[0]==p[1] && p[1]==p[2]) {
        b.rope_partial_neox(x,&positions.iter().map(|p|p[0]).collect::<Vec<_>>(),x.dim(2),rotated,theta);
        return;
    }
    let axes: Vec<_> = (0..32).map(|i| if i % 3 == 1 && i < 33 { 1 } else if i % 3 == 2 && i < 30 { 2 } else { 0 }).collect();
    rotate(b, x, positions, rotated, theta, &axes, &(0..32).collect::<Vec<_>>(), rotated);
}

pub fn vision(b: &dyn Backend, x: &mut Tensor, side: usize) {
    let width = x.dim(2);
    assert_eq!(width % 4, 0);
    let quarter = width / 4;
    let positions: Vec<_> = (0..side*side).map(|i| [0, (i / side) as u32, (i % side) as u32]).collect();
    let axes: Vec<_> = (0..width/2).map(|i| if i < quarter { 1 } else { 2 }).collect();
    let frequencies: Vec<_> = (0..width/2).map(|i| i % quarter).collect();
    rotate(b, x, &positions, width, 10000.0, &axes, &frequencies, width/2);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_positions_reduce_to_partial_neox() {
        let b = ggml_rs::default_backend();
        let mut actual = Tensor::from_vec((0..3*2*80).map(|i| (i as f32).sin()).collect(), vec![3,2,80]);
        let mut expected = actual.clone();
        b.rope_partial_neox(&mut expected, &[4,5,6], 80, 64, 10000000.0);
        text(&*b, &mut actual, &[[4;3],[5;3],[6;3]], 64, 10000000.0);
        for (a,e) in actual.data().iter().zip(expected.data()) { assert!((a-e).abs() < 1e-5); }
    }
    #[test]
    fn vision_rows_and_columns_rotate_different_dimensions() {
        let b = ggml_rs::default_backend();
        let mut x = Tensor::from_vec(vec![1.0;4*8], vec![4,1,8]);
        vision(&*b, &mut x, 2);
        assert_eq!(&x.data()[..8], &[1.0;8]);
        assert_eq!(x.data()[8], 1.0); // row zero
        assert!((x.data()[10] - (1.0f32.cos()-1.0f32.sin())).abs() < 1e-6);
        assert_eq!(x.data()[18], 1.0); // column zero
    }
}
