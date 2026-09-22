use dsv41::{ternary,formats::f16_to_f32};
use dsv41_cuda::cpu::ternary_row_kernel;

#[test]
fn ternary_dispatch_matches_scalar_with_tail_rows_and_group_scales() {
    for (n,k,r0,len) in [(19,256,2,17),(32,5120,0,32),(23,2304,4,19)] {
        let x:Vec<f32>=(0..k).map(|i| ((i%31) as f32-15.0)/32.0).collect();
        let mut seed=31u64;
        let w:Vec<u8>=(0..n*k/4).map(|_| {
            let mut b=0;
            for j in 0..4 {seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;b|=((seed%3) as u8)<<(2*j);}
            b
        }).collect();
        let s:Vec<u8>=(0..n*k/128).flat_map(|i|[0u16,0x0001,0x3800,0x4000,0x1400][i%5].to_le_bytes()).collect();
        assert!(s.chunks(2).all(|b|f16_to_f32(u16::from_le_bytes([b[0],b[1]])).is_finite()));
        let mut want=vec![0.;len];let mut got=vec![0.;len];
        ternary::rows(&x,&w,&s,k,r0,&mut want);
        ternary_row_kernel()(&x,&w,&s,k,r0,&mut got);
        assert_eq!(want,got);
    }
}
