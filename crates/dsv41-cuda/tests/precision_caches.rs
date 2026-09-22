//! Real CUDA cache isolation across ternary/original bank swaps; no model needed.
use dsv41_cuda::{Gpu, expert_cache::DeviceExpertCache};
use nrob::{ecache::Ecache, store::WeightStore, CachePolicy, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Store {record:usize, value:u8, reads:AtomicUsize}
impl WeightStore for Store {
    fn record_bytes(&self)->usize {self.record}
    fn shape(&self)->(u32,u32) {(1,1)}
    fn fetch(&self,_:u32,_:u32,dst:&mut[u8])->Result<()> {
        self.reads.fetch_add(1,Ordering::Relaxed);dst.fill(self.value);Ok(())
    }
}
#[test]
#[ignore = "requires two CUDA devices; run explicitly for the precision experiment"]
fn same_expert_key_keeps_distinct_bytes_on_both_gpus() {
    for device in [0,1] {
        let g=Gpu::new(device).expect("this local test requires both selected GPUs");
        let a=Store{record:32,value:0x55,reads:AtomicUsize::new(0)};
        let b=Store{record:64,value:0xaa,reads:AtomicUsize::new(0)};
        let (mut ha,mut hb)=(Ecache::new(128,32,CachePolicy::Lfru),Ecache::new(128,64,CachePolicy::Lfru));
        let (mut da,mut db)=(DeviceExpertCache::new(&g,2,32).unwrap(),DeviceExpertCache::new(&g,2,64).unwrap());
        for (store,len,val) in [(&a,32,0x55),(&b,64,0xaa),(&a,32,0x55)] {
            let view=da.get(&g,0,0,&ha,store).unwrap();
            let mut output=g.alloc::<u8>(len).unwrap();
            g.stream.memcpy_dtod(&view,&mut output).unwrap();
            assert_eq!(g.download(&output).unwrap(),vec![val;len]);
            g.sync().unwrap();
            std::mem::swap(&mut da,&mut db);std::mem::swap(&mut ha,&mut hb);
        }
        assert_eq!(a.reads.load(Ordering::Relaxed),1);
        assert_eq!(b.reads.load(Ordering::Relaxed),1);
    }
}
