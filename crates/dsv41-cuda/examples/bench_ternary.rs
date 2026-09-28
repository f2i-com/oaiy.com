//! Bounded, real-weight expert test. No full-model tokens/s or SSD-speed claim.
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use dsv41::{expert, ternary};
use dsv41::cpu_experts::CpuExperts;
use dsv41_cuda::{Gpu, cpu::{row_kernel,ternary_row_kernel}};
use oaiy_engine::store::WeightStore;
use oaiy_engine::ecache::Ecache;
use oaiy_engine::types::CachePolicy;
use expert::{DIM,INTER};

fn rel(a:&[f32],b:&[f32])->f64 {
    let d:f64=a.iter().zip(b).map(|(x,y)|(*x as f64-*y as f64).powi(2)).sum();
    let n:f64=b.iter().map(|x|(*x as f64).powi(2)).sum(); (d/n.max(1e-30)).sqrt()
}
fn main() {
    let args:Vec<String>=std::env::args().collect();
    let root=Path::new(args.get(1).expect("sample directory"));
    let out=Path::new(args.get(2).expect("new results directory"));
    std::fs::create_dir(out).unwrap();
    let store=ternary::TernaryStore::sample(root,40,384).unwrap();
    let cache=Ecache::new(3*ternary::RECORD_BYTES,ternary::RECORD_BYTES,CachePolicy::Lfru);
    let mut ter=Vec::new();let mut orig=Vec::new();
    for l in [0,19,39] {
        let a=cache.acquire(l,0,&store).unwrap();
        ter.push(Arc::new(a.to_vec()));
        let b=cache.acquire(l,0,&store).unwrap();assert_eq!(&*a,&*b);
        orig.push(Arc::new(std::fs::read(root.join(format!("layer{l:02}.expert000.mxfp4"))).unwrap()));
    }
    let mut missing=vec![0;ternary::RECORD_BYTES];
    assert!(store.fetch(0,1,&mut missing).is_err());
    let x:Vec<f32>=(0..DIM).map(|i| ((i%29) as f32-14.0)/16.0).collect();
    let weights=vec![0.5;3];
    let reference=CpuExperts::with_format(8,ternary::rows,ternary::RECORD_BYTES);
    let fast=CpuExperts::with_format(24,ternary_row_kernel(),ternary::RECORD_BYTES);
    let old=CpuExperts::with_kernel(24,row_kernel());
    let oracle=reference.forward(&ter,&weights,&x,10.0);
    let got=fast.forward(&ter,&weights,&x,10.0);assert_eq!(got,oracle,"AVX must match scalar exactly");
    let donor=old.forward(&orig,&weights,&x,10.0);
    for i in 0..3 { println!("expert_probe layer={} relative_output_l2={:.8}",[0,19,39][i],rel(&oracle[i],&donor[i])); }
    let g=Gpu::new(0).unwrap();
    let xd=g.upload(&x).unwrap(); let xq=g.act_quant_fp8_to(&xd.as_view()).unwrap();
    let mut results=String::from("{\n  \"full_model_benchmark\": false, \"gpu\": 0, \"threads\": 24, \"records_in_rotation\": 48,\n  \"measurements\": [\n");
    let mut first=true;
    for (label,recs,engine,is_ternary) in [("mxfp4",&orig,&old,false),("w2g128",&ter,&fast,true)] {
        // Distinct 48-record buffers exceed CPU L3 and GPU L2 for either format.
        let host:Vec<Arc<Vec<u8>>>=(0..48).map(|i|Arc::new((*recs[i%3]).clone())).collect();
        let dev:Vec<_>=host.iter().map(|r|g.upload(r).unwrap()).collect();
        let tabs:Vec<_>=(0..8).map(|b| {
            let ptrs:Vec<u64>=(0..6).map(|i|g.addr(&dev[b*6+i].as_view())).collect();
            g.upload(&Gpu::moe_table(&ptrs,&[0,1,2,3,4,5],&[0.5;6])).unwrap()
        }).collect();
        let mut h=g.alloc::<f32>(6*INTER).unwrap();let mut y=g.alloc::<f32>(6*DIM).unwrap();
        let mut forward=|batch:usize| {
            if is_ternary {g.moe_gate_up_ternary(&xq.as_view(),&tabs[batch],&mut h,6,INTER,DIM,10.0).unwrap();}
            else {g.moe_gate_up(&xq.as_view(),&tabs[batch],&mut h,6,INTER,DIM,10.0).unwrap();}
            g.act_quant_fp8(&mut h.slice_mut(..)).unwrap();
            if is_ternary {g.moe_down_ternary(&h,&tabs[batch],&mut y,6,6,INTER,DIM).unwrap();}
            else {g.moe_down(&h,&tabs[batch],&mut y,6,6,INTER,DIM).unwrap();}
        };
        forward(0);g.sync().unwrap();
        for repeat in 0..3 {
            let t=Instant::now();
            for i in 0..48 {forward(i%8);}
            g.sync().unwrap();
            let gpu_ms=t.elapsed().as_secs_f64()*1000.0/48.0;
            let _=engine.forward(&host[..6],&[0.5;6],&x,10.0);
            let t=Instant::now();
            for b in 0..8 {std::hint::black_box(engine.forward(&host[b*6..b*6+6],&[0.5;6],&x,10.0));}
            let cpu_ms=t.elapsed().as_secs_f64()*1000.0/8.0;
            println!("{label} repeat={repeat} cpu_6_experts_ms={cpu_ms:.6} gpu_6_experts_ms={gpu_ms:.6}");
            if !first {results.push_str(",\n");}first=false;
            results.push_str(&format!("    {{\"format\":\"{label}\",\"repeat\":{repeat},\"cpu_6_experts_ms\":{cpu_ms},\"gpu_6_experts_ms\":{gpu_ms}}}"));
        }
        forward(0);g.sync().unwrap();drop(forward);
        let actual=g.download(&y).unwrap();
        let expected=if is_ternary {&oracle} else {&donor};
        for i in 0..6 {
            let err=rel(&actual[i*DIM..(i+1)*DIM],&expected[i%3]);
            assert!(err<0.015,"GPU/CPU complete FFN relative error {err}");
            println!("gpu_cpu_agreement format={label} expert={} relative_l2={err:.8}",i%3);
        }
        // Separate staged H2D benchmark, allocated destination, no kernel timing.
        let mut dst=g.alloc::<u8>(host[0].len()).unwrap();
        let t=Instant::now();for i in 0..24 {g.write(&host[i],&mut dst.slice_mut(..)).unwrap();}g.sync().unwrap();
        println!("h2d format={label} ms_per_record={:.6}",t.elapsed().as_secs_f64()*1000.0/24.0);
        if is_ternary {
            // Row primitive / prefill entry: three tokens with differing inputs.
            let input:Vec<f32>=(0..3*DIM).map(|i|((i%17) as f32-8.0)/16.0).collect();
            let input_gpu=g.upload(&input).unwrap();let mut output=g.alloc::<f32>(3*INTER).unwrap();
            g.gemv_ternary(&input_gpu.as_view(),&dev[0].slice(ternary::W1),&dev[0].slice(ternary::S1),&mut output.slice_mut(..),INTER,DIM,3,false).unwrap();
            let output=g.download(&output).unwrap();
            for t in 0..3 {
                let mut cpu=vec![0.;INTER];ternary::rows(&input[t*DIM..(t+1)*DIM],&host[0][ternary::W1],&host[0][ternary::S1],DIM,0,&mut cpu);
                assert!(rel(&output[t*INTER..(t+1)*INTER],&cpu)<1e-5);
            }
            println!("three_token_ternary_row_oracle=PASS");
        }
    }
    results.push_str("\n  ],\n  \"scalar_avx_exact\": true, \"gpu_cpu_ffn_tolerance\": 0.015, \"row_oracle_tolerance\": 0.00001\n}\n");
    std::fs::write(out.join("benchmark.json"),results).unwrap();
    println!("PASS: store/cache, absent record rejection, scalar/AVX, grouped GPU and prefill row oracle");
}
