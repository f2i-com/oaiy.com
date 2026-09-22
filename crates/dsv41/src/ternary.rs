//! Experimental W2/G128 experts. This is lossy PTQ, not trained BitNet.
//! Codes: four per byte, low bits first; 0=-1, 1=0, 2=+1 (3 invalid).
//! Scales: little-endian FP16, one per row's 128 columns.
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use nrob::{Error, Result};
use nrob::store::WeightStore;
use crate::expert::{DIM, INTER};
use crate::formats::f16_to_f32;

pub const W: usize = INTER * DIM / 4;
pub const S: usize = INTER * DIM / 128 * 2;
pub const RECORD_BYTES: usize = 3 * (W + S);
pub const W1: std::ops::Range<usize> = 0..W;
pub const W2: std::ops::Range<usize> = W..2*W;
pub const W3: std::ops::Range<usize> = 2*W..3*W;
pub const S1: std::ops::Range<usize> = 3*W..3*W+S;
pub const S2: std::ops::Range<usize> = 3*W+S..3*W+2*S;
pub const S3: std::ops::Range<usize> = 3*W+2*S..RECORD_BYTES;

thread_local! { static SCRATCH: RefCell<crate::io::AlignedScratch> = RefCell::new(crate::io::AlignedScratch::default()); }
pub struct TernaryStore { root: PathBuf, layers: u32, experts: u32, direct: bool }
impl TernaryStore {
    /// Full model loading checks coverage before allocating model memory.
    pub fn open(root: &Path, source: &Path, layers: u32, experts: u32, direct: bool) -> Result<Self> {
        let store = Self::sample(root, layers, experts)?;
        if std::fs::read(root.join("source_config.json"))? != std::fs::read(source.join("config.json"))? {
            return Err(Error::Arg("ternary source configuration differs".into()));
        }
        for l in 0..layers { for e in 0..experts {
            let metadata=std::fs::metadata(store.path(l,e)).map_err(|err| Error::Arg(format!("incomplete ternary archive: expert ({l},{e}) unavailable: {err}")))?;
            if metadata.len() != RECORD_BYTES as u64 {
                return Err(Error::Arg(format!("invalid ternary record ({l},{e})")));
            }
        }}
        store.with_direct(direct)
    }
    /// Bounded samples may omit experts; fetching an absent record fails.
    pub fn sample(root: &Path, layers: u32, experts: u32) -> Result<Self> {
        if std::fs::read_to_string(root.join("format.txt"))?.trim() != "NROB_W2G128_V1" {
            return Err(Error::Arg("unsupported ternary record format".into()));
        }
        Ok(Self {root: root.to_path_buf(), layers, experts, direct:false})
    }
    pub fn with_direct(mut self, direct: bool) -> Result<Self> {
        self.direct=crate::io::open_read(&self.path(0,0),direct)?.1;
        Ok(self)
    }
    fn path(&self, l: u32, e: u32) -> PathBuf { self.root.join(format!("layer{l:02}.expert{e:03}.w2")) }
}
impl WeightStore for TernaryStore {
    fn record_bytes(&self) -> usize { RECORD_BYTES }
    fn shape(&self) -> (u32,u32) { (self.layers,self.experts) }
    fn direct_io(&self) -> bool { self.direct }
    fn fetch(&self, l: u32, e: u32, dst: &mut [u8]) -> Result<()> {
        if l >= self.layers || e >= self.experts || dst.len() != RECORD_BYTES {
            return Err(Error::Arg("ternary record bounds".into()));
        }
        let (file,direct) = crate::io::open_read(&self.path(l,e),self.direct)?;
        if file.metadata()?.len() != RECORD_BYTES as u64 { return Err(Error::Arg("ternary record length".into())); }
        if direct { SCRATCH.with(|scratch|crate::io::read_direct(&file,dst,0,&mut scratch.borrow_mut()))?; }
        else { crate::io::read_exact_at(&file,dst,0)?; }
        validate(dst)
    }
}
pub fn validate(rec: &[u8]) -> Result<()> {
    if rec.len() != RECORD_BYTES { return Err(Error::Arg("ternary record length".into())); }
    // Reject code 3 and invalid scales before exposing bytes to kernels.
    if rec[..3*W].iter().any(|&b| (b & (b >> 1) & 0x55) != 0) {
        return Err(Error::Arg("invalid ternary code 3".into()));
    }
    if rec[3*W..].chunks_exact(2).any(|b| { let s=f16_to_f32(u16::from_le_bytes([b[0],b[1]])); !s.is_finite() || s<0.0 }) {
        return Err(Error::Arg("invalid ternary scale".into()));
    }
    Ok(())
}

/// Independent portable reference, summing paired products in 32-column blocks.
pub fn rows(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
    assert!(k.is_multiple_of(128) && x.len() >= k && w.len() >= (r0+out.len())*k/4 && s.len() >= (r0+out.len())*k/64);
    for (j, y) in out.iter_mut().enumerate() {
        let r=r0+j; let mut acc=0.0;
        for b in 0..k/32 {
            let mut part=0.0;
            for p in 0..16 {
                let c=b*32+p*2;
                let q=w[r*k/4+c/4];
                let shift=(c%4)*2;
                let a=((q>>shift)&3) as i32-1;
                let z=((q>>(shift+2))&3) as i32-1;
                part += x[c]*a as f32 + x[c+1]*z as f32;
            }
            let off=(r*k/128+b/4)*2;
            acc += part*f16_to_f32(u16::from_le_bytes([s[off],s[off+1]]));
        }
        *y=acc;
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn independent_dense_oracle_and_row_offset() {
        let (n,k)=(19,256);
        let x:Vec<f32>=(0..k).map(|i| ((i%13) as f32-6.0)/8.0).collect();
        let codes:Vec<i32>=(0..n*k).map(|i| (i%3) as i32-1).collect();
        let w:Vec<u8>=codes.chunks(4).map(|c| (0..4).fold(0,|a,j| a|(((c[j]+1) as u8)<<(2*j)))).collect();
        let s=vec![0u8,0x38].repeat(n*k/128); // 0.5
        let mut got=vec![0.;17]; rows(&x,&w,&s,k,2,&mut got);
        for (j,y) in got.iter().enumerate() {
            let expected:f32=(0..k).map(|c| x[c]*codes[(j+2)*k+c] as f32*0.5).sum();
            assert_eq!(*y,expected);
        }
    }
    #[test] fn rejects_invalid_records() {
        let mut rec=vec![0x55;RECORD_BYTES]; rec[3*W..].fill(0);
        validate(&rec).unwrap(); rec[1]=3; assert!(validate(&rec).is_err());
        rec[1]=0; rec[3*W]=0; rec[3*W+1]=0x7c; assert!(validate(&rec).is_err());
        assert!(validate(&rec[..4]).is_err());
    }
}
