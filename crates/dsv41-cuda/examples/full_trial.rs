//! Fixed development-only generation comparison, exact token/timing logs.
use std::path::Path;
use std::io::Write;
use std::time::Instant;
use oaiy_engine::json::Json as J;
use dsv41::{chat,detok::Detokenizer,tokenizer::Tokenizer,model::argmax};
use dsv41_cuda::{GpuModel,GpuOptions};

fn rel(a:&[f32],b:&[f32])->f64 {
    let d:f64=a.iter().zip(b).map(|(x,y)|(*x as f64-*y as f64).powi(2)).sum();
    let n:f64=b.iter().map(|x|(*x as f64).powi(2)).sum();(d/n.max(1e-30)).sqrt()
}
fn emit(log:&mut std::fs::File,row:J) {
    let text=row.to_json();writeln!(log,"{text}").unwrap();log.flush().unwrap();println!("{text}");
}
fn finite(x:&[f32]) { assert!(x.iter().all(|v|v.is_finite()),"Nonfinite model logits"); }
fn main()->oaiy_engine::Result<()> {
    let args:Vec<String>=std::env::args().collect();
    let source=Path::new(args.get(1).expect("MODEL_DIR"));
    let out=Path::new(args.get(2).expect("FRESH_RESULT_DIR"));
    std::fs::create_dir(out)?;
    let mut log=std::fs::File::create(out.join("trial.jsonl"))?;
    let tokenizer=Tokenizer::load(source)?;let detok=Detokenizer::load(source)?;
    let suite=[
        ("capital","What is the capital of Japan? Answer with the city name only.","Tokyo"),
        ("arithmetic","What is 7 multiplied by 8? Answer with only the number.","56"),
        ("science","Complete with one word: Water freezes at zero degrees ___.","Celsius"),
        ("explanation","Explain why rainbows form in two short sentences.","")
    ];
    // Freeze prompts before evaluating either checkpoint. No adaptation uses these.
    let prompts:Vec<Vec<u32>>=suite.iter().map(|(_,text,_)| {
        let user=J::obj([("role",J::str("user")),("content",J::str(*text))]);
        tokenizer.encode(&chat::encode(&[user],&chat::Options::default()).unwrap().prompt)
    }).collect();
    let opts=GpuOptions {devices:vec![0,1],max_seq:512,expert_cache_bytes:140<<30,direct_io:true,
        vram_expert_bytes:Some(16<<30),vram_headroom_bytes:1<<30,cpu_expert_threads:Some(24),vision:false,residual_on_device:None};
    let t=Instant::now();let mut model=GpuModel::load(source,&source.join("engram_meta.safetensors"),&opts)?;
    emit(&mut log,J::obj([("event",J::str("loaded")),("seconds",J::Num(t.elapsed().as_secs_f64())),
        ("source",J::str(source.display().to_string())),("ram_cache_bytes",J::Int(140i64<<30)),
        ("vram_cache_bytes_per_gpu",J::Int(16i64<<30)),("gpus",J::Arr(vec![J::Int(0),J::Int(1)])),("max_new_tokens",J::Int(32))]));
    let speed_user=J::obj([("role",J::str("user")),("content",J::str("Explain how rainbows form."))]);
    let speed_prefix=tokenizer.encode(&chat::encode(&[speed_user],&chat::Options::default())?.prompt);
    let prose=" Rainbows occur when sunlight passes through water droplets in the air. Refraction changes the direction of light, and dispersion separates its colors. Some light reflects inside each droplet before leaving it. Different colors leave at different angles, so an observer sees a curved band of colors. ";
    let speed_tokens=tokenizer.encode(&prose.repeat(3));assert!(speed_tokens.len()>=64);
    for pass in 0..2 {
        let host0=model.expert_cache().stats();let t=Instant::now();
        finite(&model.forward(&speed_prefix,0)?);let prefill=t.elapsed().as_secs_f64();
        let mut times=Vec::new();
        for (i,token) in speed_tokens[..64].iter().enumerate() {
            let t=Instant::now();finite(&model.forward(&[*token],speed_prefix.len()+i)?);times.push(t.elapsed().as_secs_f64());
        }
        let host=model.expert_cache().stats();let total:f64=times.iter().sum();
        emit(&mut log,J::obj([("event",J::str("forced_speed")),("pass",J::Int(pass)),
            ("prefix_ids",J::Arr(speed_prefix.iter().map(|v|J::Int(*v as i64)).collect())),
            ("input_ids",J::Arr(speed_tokens[..64].iter().map(|v|J::Int(*v as i64)).collect())),
            ("prefill_seconds",J::Num(prefill)),("forward_seconds",J::Arr(times.into_iter().map(J::Num).collect())),
            ("tokens_per_second",J::Num(64.0/total)),("ram_hits",J::Int((host.hits-host0.hits) as i64)),
            ("ssd_fetches",J::Int((host.misses-host0.misses) as i64))]));
    }
    for pass in 0..2 {for (idx,ids) in prompts.iter().enumerate() {
        let host0=model.expert_cache().stats();
        let t=Instant::now();let logits=model.forward(ids,0)?;finite(&logits);
        let prefill=t.elapsed().as_secs_f64();let mut token=argmax(&logits);
        let mut generated=Vec::new();let mut bytes=Vec::new();let mut times=Vec::new();
        for step in 0..32 {
            generated.push(token);bytes.extend_from_slice(detok.bytes(token));
            if token==model.cfg.eos_token_id || step==31 {break;}
            let t=Instant::now();let logits=model.forward(&[token],ids.len()+step)?;finite(&logits);
            times.push(t.elapsed().as_secs_f64());token=argmax(&logits);
        }
        let total:f64=times.iter().sum();let host=model.expert_cache().stats();
        emit(&mut log,J::obj([("event",J::str("answer")),("pass",J::Int(pass)),("id",J::str(suite[idx].0)),
            ("prompt",J::str(suite[idx].1)),("expected",J::str(suite[idx].2)),
            ("prompt_ids",J::Arr(ids.iter().map(|v|J::Int(*v as i64)).collect())),
            ("generated_ids",J::Arr(generated.iter().map(|v|J::Int(*v as i64)).collect())),
            ("text",J::str(String::from_utf8_lossy(&bytes).to_string())),("prefill_seconds",J::Num(prefill)),
            ("decode_forward_seconds",J::Arr(times.iter().map(|v|J::Num(*v)).collect())),
            ("decode_forwards",J::Int(times.len() as i64)),
            ("decode_tokens_per_second",if total>0. {J::Num(times.len() as f64/total)} else {J::Null}),
            ("ram_hits",J::Int((host.hits-host0.hits) as i64)),("ssd_fetches",J::Int((host.misses-host0.misses) as i64))]));
    }}
    // Same retained architecture, new expert arithmetic: compare chunking/reset/snapshot.
    let ids=&prompts[0];let cut=(ids.len()/2).max(2).min(ids.len()-2);
    let whole=model.forward(ids,0)?;finite(&whole);
    model.forward(&ids[..cut],0)?;let snap=model.snapshot(cut)?;
    let chunked=model.forward(&ids[cut..],cut)?;finite(&chunked);
    let chunk_error=rel(&whole,&chunked);
    assert!(chunk_error<1e-3,"Chunked prefill disagrees: {chunk_error}");
    model.forward(&prompts[1],0)?;
    model.restore_snapshot(&snap,&ids[..cut])?;
    let restored=model.forward(&ids[cut..],cut)?;let snapshot_error=rel(&chunked,&restored);
    assert!(snapshot_error<1e-4,"Snapshot replay disagrees: {snapshot_error}");
    let reset=model.forward(ids,0)?;let reset_error=rel(&whole,&reset);
    assert!(reset_error<1e-4,"Reset disagrees: {reset_error}");
    emit(&mut log,J::obj([("event",J::str("checks")),("chunk_relative_l2",J::Num(chunk_error)),
        ("snapshot_relative_l2",J::Num(snapshot_error)),("reset_relative_l2",J::Num(reset_error)),("passed",J::Bool(true))]));
    Ok(())
}
