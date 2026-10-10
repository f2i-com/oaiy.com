//! `media`'s tests.

use super::*;

fn cfg(extra_image: &str, extra_video: &str) -> Json {
    let mut v = config::default_json();
    let image = Json::parse(format!(r#"{{"enabled":true,"default_model":"qwen","memory":"auto","ram_gb":32,"vram_gb":20,"models":{{
            "qwen":{{"architecture":"qwen-image","base":"base","transformer":"q.gguf","adapter":"turbo.safetensors"{extra_image}}},
            "anime":{{"architecture":"sdxl","checkpoint":"/abs/anime.safetensors","tokenizer":"tok.json","steps":24,"memory":"ssd"}}}}}}"#).as_bytes()).unwrap();
    let video = Json::parse(format!(r#"{{"enabled":true,"default_model":"sulphur","ram_gb":48,"vram_gb":null,"fps":24,"ffmpeg":"ffmpeg","models":{{
            "sulphur":{{"family":"sulphur-2","transformer":"s.safetensors","vae":"s.safetensors","text_encoder":"g.safetensors","tokenizer":"t.json"{extra_video}}}}}}}"#).as_bytes()).unwrap();
    let Json::Obj(top) = &mut v else { unreachable!() };
    let media = &mut top.iter_mut().find(|(k, _)| k == "media").unwrap().1;
    crate::util::set(media, "image", image);
    crate::util::set(media, "video", video);
    crate::util::set(media, "device", Json::Int(1));
    config::validate(&v).unwrap();
    v
}

fn body(s: &str) -> Json {
    Json::parse(s.as_bytes()).unwrap()
}

#[test]
fn klein_requests_use_native_paths_fixed_distilled_recipe_and_optional_style_loras() {
    let mut c=cfg("","");
    let m=body(r#"{"architecture":"flux2-klein-4b","transformer":"klein.safetensors","text_encoder":"qwen_3_4b.safetensors","vae":"flux2-vae.safetensors","tokenizer":"qwen-tokenizer.json","loras":[{"path":"style.safetensors","strength":0.75}]}"#);
    let mut media=c.get("media").unwrap().clone();
    let mut image=media.get("image").unwrap().clone();
    let mut models=image.get("models").unwrap().clone();
    crate::util::set(&mut models,"klein",m);
    crate::util::set(&mut image,"models",models);
    crate::util::set(&mut media,"image",image);
    crate::util::set(&mut c,"media",media);
    config::validate(&c).unwrap();
    let root=Path::new("/install");
    let (r,..)=image_request(&c,root,Path::new("/install/outputs"),&body(r#"{"model":"klein","prompt":"a fox","width":512,"height":512}"#),false).unwrap();
    assert_eq!(r.get("architecture").and_then(Json::as_str),Some("flux2-klein-4b"));
    assert_eq!(r.get("steps").and_then(Json::as_i64),Some(4));
    assert_eq!(r.get("cfg").and_then(Json::as_f64),Some(1.0));
    assert_eq!(r.get("loras").and_then(Json::as_array).unwrap().len(),1);
    assert!(r.get("base").is_none()&&r.get("adapter").is_none());
    let (baseline,..)=image_request(&c,root,root,&body(r#"{"model":"klein","prompt":"a fox","use_loras":false}"#),false).unwrap();
    assert!(baseline.get("loras").is_none());
    let (empty_negative,..)=image_request(&c,root,root,&body(r#"{"model":"klein","prompt":"a fox","size":"512x512","negative_prompt":" ","images":[]}"#),false).unwrap();
    assert_eq!(empty_negative.get("width").and_then(Json::as_i64),Some(512));
    assert_eq!(empty_negative.get("height").and_then(Json::as_i64),Some(512));
    assert!(empty_negative.get("negative_prompt").is_none());
    for bad in [r#"{"model":"klein","prompt":"x","steps":8}"#,r#"{"model":"klein","prompt":"x","cfg":4}"#,r#"{"model":"klein","prompt":"x","image":"data:image/png;base64,a"}"#,r#"{"model":"klein","prompt":"x","negative_prompt":"blur"}"#] {
        assert!(image_request(&c,root,root,&body(bad),false).is_err(),"{bad}");
    }
}

#[test]
fn image_requests_map_openai_fields_onto_the_worker() {
    let c = cfg("", "");
    let root = Path::new("/install");
    let out = Path::new("/install/outputs");
    let (r, name, size, n) = image_request(&c, root, out, &body(r#"{"prompt":"a fox","n":2,"size":"1536x1024","model":"dall-e-3","seed":5}"#), false).unwrap();
    assert_eq!((name.as_str(), size.as_str(), n), ("qwen", "1536x1024", 2));
    assert_eq!(r.get("steps").and_then(Json::as_i64), Some(6));
    assert_eq!(r.get("device").and_then(Json::as_i64), Some(1));
    assert_eq!(r.get("transformer").and_then(Json::as_str), Some(root.join("q.gguf").to_str().unwrap()));
    assert_eq!(r.get("memory").and_then(Json::as_str), Some("auto"));
    assert_eq!((r.get("ram_gb").and_then(Json::as_i64), r.get("vram_gb").and_then(Json::as_i64)), (Some(32), Some(20)));
    assert!(r.get("output_dir").and_then(Json::as_str).unwrap().contains("images"));
    // The SDXL model's own residency applies; requests may lower but not raise caps.
    let (s, ..) = image_request(&c, root, out, &body(r#"{"prompt":"x","model":"anime","size":"1000x1000","ram_gb":8}"#), false).unwrap();
    assert_eq!(s.get("architecture").and_then(Json::as_str), Some("sdxl"));
    assert_eq!((s.get("width").and_then(Json::as_i64), s.get("steps").and_then(Json::as_i64)), (Some(960), Some(24)));
    assert_eq!(s.get("memory").and_then(Json::as_str), Some("ssd"));
    assert_eq!(s.get("ram_gb").and_then(Json::as_i64), Some(8));
    for bad in [r#"{"prompt":""}"#, r#"{"prompt":"x","n":17}"#, r#"{"prompt":"x","size":"big"}"#, r#"{"prompt":"x","ram_gb":33}"#,
        r#"{"prompt":"x","memory":"disk"}"#, r#"{"prompt":"x","model":"missing"}"#, r#"{"prompt":"x","steps":5}"#, r#"{"prompt":"x","size":"4096x4096"}"#] {
        assert!(image_request(&c, root, out, &body(bad), false).is_err(), "{bad}");
    }
}

#[test]
fn edits_carry_references_and_take_their_shape() {
    let c = cfg("", "");
    let out = std::env::temp_dir().join(format!("oaiy-studio-edit-{}", std::process::id()));
    let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
    png.extend(1920u32.to_be_bytes());
    png.extend(1080u32.to_be_bytes());
    let url = format!("data:image/png;base64,{}", crate::util::base64_encode(&png));
    let edit = body(&format!(r#"{{"prompt":"make it night","images":[{{"image_url":"{url}"}}]}}"#));
    let (r, _, size, _) = image_request(&c, Path::new("/install"), &out, &edit, false).unwrap();
    assert_eq!(size, "1376x768");
    let refs = r.get("images").and_then(Json::as_array).unwrap();
    assert_eq!(std::fs::read(refs[0].as_str().unwrap()).unwrap(), png);
    let four = body(&format!(r#"{{"prompt":"x","image":["{url}","{url}","{url}","{url}"]}}"#));
    assert!(image_request(&c, Path::new("/install"), &out, &four, false).unwrap_err().contains("three"));
    let sdxl = body(&format!(r#"{{"prompt":"x","model":"anime","image":"{url}"}}"#));
    assert!(image_request(&c, Path::new("/install"), &out, &sdxl, false).unwrap_err().contains("SDXL"));
    assert!(image_request(&c, Path::new("/install"), &out, &body(r#"{"prompt":"x","image":"C:/private.png"}"#), false).is_err());
    let _ = std::fs::remove_dir_all(out);
}

#[test]
fn gpu_faults_are_told_from_bad_requests() {
    assert!(gpu_fault("DriverError(CUDA_ERROR_LAUNCH_FAILED, \"unspecified launch failure\")"));
    assert!(gpu_fault("nonfinite decoded video pixels"));
    assert!(gpu_fault("wgpu error: Validation Error: Parent device is lost"));
    assert!(gpu_fault("Device lost (DeviceLostReason::Unknown): the GPU driver was reset"));
    assert!(!gpu_fault("size must look like 1024x1024"));
}

#[test]
fn video_requests_fit_openai_sizes_and_seconds_to_ltx() {
    let c = cfg("", "");
    let root = Path::new("/install");
    let out = std::env::temp_dir().join(format!("oaiy-studio-video-{}", std::process::id()));
    let (r, name, size, secs) = video_request(&c, root, &out, &body(r#"{"prompt":"waves","model":"sora-2","seconds":"4","size":"1280x720"}"#), false).unwrap();
    assert_eq!((name.as_str(), size.as_str()), ("sulphur", "1024x576"));
    assert_eq!(r.get("model").and_then(Json::as_str), Some("sulphur-2"));
    assert_eq!(r.get("frames").and_then(Json::as_i64), Some(97));
    assert!((secs - 4.0).abs() < 1e-9);
    assert_eq!(r.get("vram_gb").and_then(Json::as_i64), Some(192));
    assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(48));
    // A data: URL reference becomes a file the worker reads; local paths need trust.
    let png = format!(r#"{{"prompt":"x","input_reference":{{"image_url":"data:image/png;base64,{}"}}}}"#, crate::util::base64_encode(b"\x89PNG"));
    let (r, ..) = video_request(&c, root, &out, &body(&png), false).unwrap();
    assert_eq!(std::fs::read(r.get("image").and_then(Json::as_str).unwrap()).unwrap(), b"\x89PNG");
    assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","image":"C:/secret.png"}"#), false).is_err());
    assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","input_reference":{"file_id":"f"}}"#), false).is_err());
    let (long, ..) = video_request(&c, root, &out, &body(r#"{"prompt":"x","seconds":30}"#), false).unwrap();
    assert_eq!(long.get("frames").and_then(Json::as_i64), Some(121));
    // With no size asked for, a start frame gives the clip its shape (a
    // 512x768 portrait stays portrait); a size asked for still wins.
    let mut header = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
    header.extend(512u32.to_be_bytes());
    header.extend(768u32.to_be_bytes());
    let portrait = crate::util::base64_encode(&header);
    let (r, _, size, _) = video_request(&c, root, &out, &body(&format!(r#"{{"prompt":"x","input_reference":{{"image_url":"data:image/png;base64,{portrait}"}}}}"#)), false).unwrap();
    assert_eq!(size, "512x768");
    assert_eq!((r.get("width").and_then(Json::as_i64), r.get("height").and_then(Json::as_i64)), (Some(512), Some(768)));
    let (_, _, size, _) = video_request(&c, root, &out, &body(&format!(r#"{{"prompt":"x","size":"768x512","input_reference":{{"image_url":"data:image/png;base64,{portrait}"}}}}"#)), false).unwrap();
    assert_eq!(size, "768x512");
    // A request's negative prompt and NAG settings go to the worker.
    let (r, ..) = video_request(&c, root, &out, &body(r#"{"prompt":"x","negative_prompt":"watermark","nag":{"tau":3.5}}"#), false).unwrap();
    assert_eq!(r.get("negative_prompt").and_then(Json::as_str), Some("watermark"));
    assert_eq!(r.get("nag").and_then(|n| n.get("tau")).and_then(Json::as_f64), Some(3.5));
    let (r, ..) = video_request(&c, root, &out, &body(r#"{"prompt":"x"}"#), false).unwrap();
    assert!(r.get("negative_prompt").is_none());
    let _ = std::fs::remove_dir_all(out);
}

#[test]
fn video_follows_a_soundtrack_or_speech() {
    let mut c = cfg("", r#","audio_vae":"s.safetensors""#);
    let speech = Json::parse(br#"{"enabled":true,"default_model":"tts","voices_dir":"voices","models":{"tts":{"design":"/m/design","base":"/m/base"}}}"#).unwrap();
    let Json::Obj(top) = &mut c else { unreachable!() };
    let media = &mut top.iter_mut().find(|(k, _)| k == "media").unwrap().1;
    crate::util::set(media, "speech", speech);
    let root = Path::new("/install");
    let out = std::env::temp_dir().join(format!("oaiy-studio-a2v-{}", std::process::id()));
    // OpenAI's audio input shape: the file is written, the length comes from it.
    let wav = format!(r#"{{"prompt":"a woman talks","input_audio":{{"data":"{}","format":"wav"}}}}"#, crate::util::base64_encode(b"RIFF"));
    let (r, ..) = video_request(&c, root, &out, &body(&wav), false).unwrap();
    assert_eq!(std::fs::read(r.get("audio_file").and_then(Json::as_str).unwrap()).unwrap(), b"RIFF");
    assert!(r.get("frames").is_none(), "the worker sizes the clip to the soundtrack");
    // An explicit length still wins.
    let (r, ..) = video_request(&c, root, &out, &body(&wav.replace(r#""prompt""#, r#""seconds":2,"prompt""#)), false).unwrap();
    assert_eq!(r.get("frames").and_then(Json::as_i64), Some(49));
    // Speech is made first, in the same job, with a described voice.
    let (r, _, _, secs) = video_request(&c, root, &out, &body(r#"{"prompt":"a man speaks to camera","speech":{"text":"Hello there, welcome back.","voice":"onyx"}}"#), false).unwrap();
    let s = r.get("speech").unwrap();
    assert_eq!((str_or(s, "kind", ""), str_or(s, "text", "")), ("speech", "Hello there, welcome back."));
    assert!(secs > 0.5 && secs <= 5.0, "{secs}");
    // Refusals: no audio VAE, audio off, both sources, a local file from afar.
    assert!(video_request(&cfg("", ""), root, &out, &body(&wav), false).unwrap_err().contains("audio VAE"));
    assert!(video_request(&c, root, &out, &body(&wav.replace(r#""prompt""#, r#""audio":false,"prompt""#)), false).is_err());
    assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","input_audio":"C:/voice.wav"}"#), false).unwrap_err().contains("this machine"));
    // Two stages when asked, with the dev weights and upsampler set: the dev
    // transformer guides at half size, the configured one refines.
    assert!(r.get("guidance").is_none(), "no dev weights: one distilled pass");
    let two = cfg("", r#","audio_vae":"s.safetensors","dev_transformer":"dev.safetensors","spatial_upscaler":"up.safetensors""#);
    let (r, _, size, _) = video_request(&two, root, &out, &body(&wav.replace(r#""prompt""#, r#""pipeline":"two_stage","size":"1000x560","prompt""#)), false).unwrap();
    assert_eq!(size, "1024x576", "sides in steps of 64");
    assert!(r.get("transformer").and_then(Json::as_str).unwrap().ends_with("dev.safetensors"));
    assert!(r.get("guidance").is_some());
    let refine = r.get("refine").unwrap();
    assert!(str_or(refine, "upsampler", "").ends_with("up.safetensors"));
    assert!(!str_or(refine, "transformer", "").ends_with("dev.safetensors"));
    let (fast, ..) = video_request(&two, root, &out, &body(&wav), false).unwrap();
    assert!(fast.get("guidance").is_none() && fast.get("refine").is_none(), "auto: one distilled pass");
    assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","pipeline":"two_stage"}"#), false).unwrap_err().contains("dev_transformer"));
    // Lip-synced speech with the model's ID-LoRA: the saved voice's sample is the reference.
    let mut id = cfg("", r#","audio_vae":"s.safetensors","id_lora":"id.safetensors""#);
    let Json::Obj(top) = &mut id else { unreachable!() };
    let media = &mut top.iter_mut().find(|(k, _)| k == "media").unwrap().1;
    crate::util::set(media, "speech", Json::parse(format!(r#"{{"enabled":true,"default_model":"tts","voices_dir":"{}","models":{{"tts":{{"design":"/m/design","base":"/m/base"}}}}}}"#, out.join("voices").to_string_lossy().replace('\\', "/")).as_bytes()).unwrap());
    std::fs::create_dir_all(out.join("voices")).unwrap();
    std::fs::write(out.join("voices").join("gary.json"), br#"{"name":"Gary"}"#).unwrap();
    std::fs::write(out.join("voices").join("gary.wav"), b"RIFF").unwrap();
    let (r, ..) = video_request(&id, root, &out, &body(r#"{"prompt":"a man speaks","speech":{"text":"We are closing.","voice":"Gary"}}"#), false).unwrap();
    assert_eq!(r.get("identity").and_then(Json::as_bool), Some(true));
    assert!(str_or(r.get("lora").unwrap(), "path", "").ends_with("id.safetensors"));
    assert!(r.get("reference_voice").and_then(Json::as_str).unwrap().ends_with("gary.wav"));
    // A speech file is kept and followed; with lip_sync voice it is the voice instead.
    let (r, ..) = video_request(&id, root, &out, &body(&wav.replace(r#""prompt""#, r#""transcript":"We are closing.","prompt""#)), false).unwrap();
    assert!(r.get("identity").is_none() && r.get("audio_file").is_some());
    let (r, ..) = video_request(&id, root, &out, &body(&wav.replace(r#""prompt""#, r#""lip_sync":"voice","transcript":"We are closing.","prompt""#)), false).unwrap();
    assert!(r.get("identity").is_some() && r.get("reference_voice").is_none());
    let (r, ..) = video_request(&id, root, &out, &body(&wav.replace(r#""prompt""#, r#""soundtrack_mode":"frozen","prompt""#)), false).unwrap();
    assert_eq!(r.get("soundtrack_mode").and_then(Json::as_str), Some("frozen"));
    // Without words, or with lip_sync off, the audio is followed as a soundtrack.
    let (r, ..) = video_request(&id, root, &out, &body(&wav), false).unwrap();
    assert!(r.get("identity").is_none());
    let (r, ..) = video_request(&id, root, &out, &body(r#"{"prompt":"a man speaks","lip_sync":"off","speech":{"text":"We are closing.","voice":"Gary"}}"#), false).unwrap();
    assert!(r.get("identity").is_none() && r.get("lora").is_none());
    assert!(video_request(&c, root, &out, &body(r#"{"prompt":"x","input_audio":"data:audio/wav;base64,UklGRg==","speech":{"text":"hi"}}"#), false).is_err());
    let _ = std::fs::remove_dir_all(out);
}

#[test]
fn sizes_fit_and_urls_stay_inside_the_output_root() {
    assert_eq!(fit(1792, 1024, 1024, 128, 32), (1024, 576));
    assert_eq!(fit(720, 1280, 1024, 128, 32), (576, 1024));
    assert_eq!(fit(512, 320, 1024, 128, 32), (512, 320));
    let root = std::env::temp_dir().join(format!("oaiy-studio-media-files-{}", std::process::id()));
    std::fs::create_dir_all(root.join("images")).unwrap();
    let file = root.join("images").join("a b.png");
    std::fs::write(&file, b"x").unwrap();
    assert_eq!(file_url(&root, &file).as_deref(), Some("/files/images/a b.png"));
    assert_eq!(resolve_file(&root, "images/a%20b.png"), Some(file.canonicalize().unwrap()));
    assert!(resolve_file(&root, "../secret").is_none());
    assert!(resolve_file(&root, "images/../../x").is_none());
    assert!(resolve_file(&root, "C:/Windows/win.ini").is_none());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn worker_events_become_monotonic_progress() {
    let e = |s: &str| body(s);
    let (p, stage) = progress_of(Kind::Image, 2, &e(r#"{"stage":"sampling","image":2,"step":3,"steps":6}"#)).unwrap();
    assert_eq!(stage, "sampling");
    assert!((p - 75.0).abs() < 1e-9);
    let (p, _) = progress_of(Kind::Video, 1, &e(r#"{"stage":"video_denoising","current":192,"total":384}"#)).unwrap();
    assert!((p - 51.5).abs() < 1e-9);
    assert!(progress_of(Kind::Image, 1, &e(r#"{"error":"x"}"#)).is_none());
}

#[test]
fn jobs_queue_cancel_and_trim_to_the_kept_count() {
    let m = Media::new();
    let a = m.submit(Kind::Image, body(r#"{"prompt":"a"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 2, None);
    assert!(m.busy());
    assert!(m.cancel(&a.id));
    assert_eq!(m.get(&a.id).unwrap().status, "cancelled");
    m.submit(Kind::Image, body(r#"{"prompt":"b"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 2, None);
    // Just finished: a synchronous caller may still be reading it, so it stays.
    assert!(m.get(&a.id).is_some());
    m.update(&a.id, |j| j.completed_at = Some(0));
    m.submit(Kind::Video, body(r#"{"prompt":"c"}"#), "v".into(), "1x1".into(), 1, 1.0, 2, None);
    assert!(m.get(&a.id).is_none(), "a settled finished job made room");
    assert_eq!(m.list().len(), 2);
    assert_eq!(m.wait("missing", Duration::ZERO).map(|j| j.id), None);
}

#[test]
fn incognito_jobs_stay_out_of_lists_and_leave_nothing_behind() {
    let m = Media::new();
    let dir = std::env::temp_dir().join(format!("oaiy-studio-incognito-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::write(dir.join("images").join("a.png"), b"x").unwrap();
    let job = m.submit(Kind::Image, body(r#"{"prompt":"secret"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 10, Some(dir.clone()));
    assert!(job.incognito);
    assert!(m.list().is_empty(), "incognito jobs are not listed");
    assert!(m.get(&job.id).is_some(), "but their owner can poll them by id");
    let private = m.private_activity();
    assert_eq!(private.len(), 1, "the UI still learns that something runs");
    assert!(private.at(0).unwrap().get("prompt").is_none(), "but not what");
    // Cancelled while queued, it never reached the runner that stamps an
    // expiry; the next sweep takes it anyway.
    m.cancel(&job.id);
    assert!(m.get(&job.id).unwrap().expires_at.is_none());
    m.sweep();
    assert!(m.get(&job.id).is_none());
    assert!(!dir.exists(), "its folder is gone");
    // Deleting a running job forgets it once it stops.
    let run = m.submit(Kind::Image, body(r#"{"prompt":"x"}"#), "qwen".into(), "1x1".into(), 1, 0.0, 10, None);
    m.update(&run.id, |j| j.status = "in_progress".into());
    assert!(m.remove(&run.id));
    m.update(&run.id, |j| j.status = "cancelled".into());
    m.sweep();
    assert!(m.get(&run.id).is_none());
}

#[test]
fn a_model_on_gpus_of_its_own_is_paused_for_media_there() {
    let cfg = |models: &str| Json::parse(format!(r#"{{"media":{{"device":1}},"llm":{{"devices":[0],"models":{models}}}}}"#).as_bytes()).unwrap();
    assert!(!pauses_llm(&cfg(r#"[{"name":"a"}]"#)));
    assert!(pauses_llm(&cfg(r#"[{"name":"a"},{"name":"big","devices":[0,1]}]"#)));
    // A disabled one does not count.
    assert!(!pauses_llm(&cfg(r#"[{"name":"big","devices":[0,1],"enabled":false}]"#)));
}

#[test]
fn only_models_on_the_media_gpu_take_turns_with_media() {
    let cfg = |policy: &str| Json::parse(format!(r#"{{"media":{{"device":1,"llm_policy":"{policy}"}},"llm":{{"devices":[0],"default_model":"small",
            "models":[{{"name":"small"}},{{"name":"big","devices":[0,1]}},{{"name":"off","devices":[1],"enabled":false}}]}}}}"#).as_bytes()).unwrap();
    let auto = cfg("auto");
    assert_eq!(model_devices(&auto, "small"), [0]);
    assert_eq!(model_devices(&auto, "big"), [0, 1]);
    assert!(!needs_media_gpu(&auto, Some("small")), "GPU 0 keeps answering");
    assert!(needs_media_gpu(&auto, Some("big")));
    assert!(!needs_media_gpu(&auto, None), "the default model is on GPU 0");
    assert!(!needs_media_gpu(&auto, Some("off")), "a disabled model is not served: the default is");
    assert_eq!(resolve_model(&auto, Some("nope")).as_deref(), Some("small"));
    assert!(needs_media_gpu(&cfg("pause_llm"), Some("small")));
    assert!(!needs_media_gpu(&cfg("coexist"), Some("big")));
    // No LLM GPUs set: oaiy-llm-server's first two.
    let both = Json::parse(br#"{"media":{"device":1},"llm":{"devices":[],"models":[{"name":"a"}]}}"#).unwrap();
    assert!(needs_media_gpu(&both, None));
}

#[test]
fn chat_leases_wait_for_exclusive_media_and_time_out() {
    let m = Media::new();
    {
        let _a = m.chat_lease(true, Duration::ZERO).unwrap();
        assert_eq!(m.broker.lock().unwrap().chats, 1);
    }
    assert_eq!(m.broker.lock().unwrap().chats, 0);
    {
        let _b = m.chat_lease(false, Duration::ZERO).unwrap();
        assert_eq!(m.broker.lock().unwrap().chats, 0, "a chat on another GPU never holds media up");
    }
    m.broker.lock().unwrap().media = true;
    assert!(m.chat_lease(true, Duration::from_millis(20)).is_err());
    assert!(m.chat_lease(false, Duration::ZERO).is_ok(), "coexisting devices never wait");
}

#[test]
fn qwen_image_turbo_takes_its_own_eight_steps_at_cfg_one_and_no_adapter() {
    let mut c = cfg("", "");
    let turbo = body(r#"{"architecture":"qwen-image","base":"base","transformer":"Qwen-Image-2.1-Turbo-AD-Q4_K.gguf","text_encoder":"te.gguf","distilled":true}"#);
    let mut models = c.get("media").and_then(|m| m.get("image")).and_then(|i| i.get("models")).unwrap().clone();
    crate::util::set(&mut models, "turbo", turbo);
    crate::util::set(crate::registry::obj_mut(&mut c, &["media", "image"]).unwrap(), "models", models);
    config::validate(&c).unwrap();
    let root = Path::new("/install");
    let (r, ..) = image_request(&c, root, root, &body(r#"{"model":"turbo","prompt":"a fox"}"#), false).unwrap();
    assert_eq!(r.get("steps").and_then(Json::as_i64), Some(8));
    assert_eq!(r.get("distilled"), Some(&Json::Bool(true)));
    assert!(r.get("cfg").is_none(), "the worker's own, 1");
    assert_eq!(r.get("adapter"), Some(&Json::Null));
    assert_eq!(r.get("text_encoder").and_then(Json::as_str), Some("/install/te.gguf"));
    let err = image_request(&c, root, root, &body(r#"{"model":"turbo","prompt":"a fox","steps":30}"#), false).unwrap_err();
    assert!(err.contains("8 steps"), "{err}");
    // An adapter beside a distilled checkpoint is refused where it is set.
    let image = crate::registry::obj_mut(&mut c, &["media", "image", "models", "turbo"]).unwrap();
    crate::util::set(image, "adapter", Json::str("turbo.safetensors"));
    assert!(config::validate(&c).unwrap_err().contains("no turbo adapter"));
}

#[test]
fn qwen_image_pictures_go_to_the_egpu_where_it_runs_oaiys_engine() {
    let mut c = cfg("", "");
    let llm = crate::registry::obj_mut(&mut c, &["llm"]).unwrap();
    crate::util::set(llm, "egpu", body(r#"{"enabled":true,"engine":"webgpu"}"#));
    let qwen = body(r#"{"base":"b","transformer":"t.gguf","steps":8}"#);
    assert!(on_card(&c, Kind::Image, &qwen));
    assert!(!on_card(&c, Kind::Video, &qwen), "pictures only");
    assert!(!on_card(&c, Kind::Image, &body(r#"{"architecture":"sdxl","checkpoint":"c"}"#)), "Qwen Image only");
    assert!(!on_card(&c, Kind::Image, &body(r#"{"base":"b","backend":"cpu"}"#)));
    for egpu in [r#"{"enabled":true,"engine":"webgpu","images":false}"#, r#"{"enabled":true,"engine":"tinygrad"}"#, r#"{"enabled":false,"engine":"webgpu"}"#] {
        crate::util::set(crate::registry::obj_mut(&mut c, &["llm"]).unwrap(), "egpu", body(egpu));
        assert!(!on_card(&c, Kind::Image, &qwen), "{egpu}");
    }
}
