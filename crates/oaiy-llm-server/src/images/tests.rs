//! `images`'s tests.

use super::*;
#[test]
fn sdxl_catalog_routes_checkpoint_and_validates_before_handoff() {
    let root = std::env::temp_dir().join(format!("oaiy-sdxl-catalog-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let fixture = root.join("checkpoint.safetensors");
    std::fs::write(&fixture, b"fixture").unwrap();
    let write = |name: &str, j: Json| std::fs::write(root.join(name), j.to_json()).unwrap();
    write("controller.json", Json::obj([
        ("worker", Json::str("checkpoint.safetensors")), ("controller_path", Json::str("checkpoint.safetensors")),
        ("controller_name", Json::str("qwen")), ("controller_device", Json::Int(0)),
        ("image_device", Json::Int(1)), ("output_root", Json::str("outputs")),
    ]));
    write("image.json", Json::obj([
        // Missing shared Qwen files must not prevent selecting SDXL.
        ("base", Json::str("missing-qwen")), ("adapter", Json::str("missing-adapter")),
        ("default_model", Json::str("anime")),
        ("models", Json::obj([("anime", Json::obj([
            ("architecture", Json::str("sdxl")), ("checkpoint", Json::str("checkpoint.safetensors")),
            ("tokenizer", Json::str("checkpoint.safetensors")), ("steps", Json::Int(20)), ("cfg", Json::Num(3.0)),
        ]))])),
    ]));
    let cfg = Config::read(&root).unwrap();
    let request = |extra: &str| Json::parse(format!(r#"{{"prompt":"anime fox"{extra}}}"#).as_bytes()).unwrap();
    let prepared = prepare(&cfg, &request("")).unwrap();
    assert_eq!(prepared.get("architecture").and_then(Json::as_str), Some("sdxl"));
    assert_eq!(prepared.get("steps").and_then(Json::as_i64), Some(20));
    assert_eq!(prepared.get("cfg").and_then(Json::as_f64), Some(3.0));
    assert!(prepared.get("adapter").is_none());
    assert_eq!(prepared.get("device").and_then(Json::as_i64), Some(1));
    let overridden = prepare(&cfg, &request(r#", "steps":16,"cfg":2.5,"negative_prompt":"blurry","checkpoint":"untrusted.safetensors""#)).unwrap();
    assert_eq!(overridden.get("steps").and_then(Json::as_i64), Some(16));
    assert_eq!(overridden.get("checkpoint"), prepared.get("checkpoint"));
    assert_eq!(overridden.get("negative_prompt").and_then(Json::as_str), Some("blurry"));
    assert_eq!(cfg.capabilities().get("image").unwrap().get("available").and_then(Json::as_bool), Some(true));
    for extra in [r#", "width":544"#, r#", "cfg":0"#, r#", "cfg":"2.5""#, r#", "turbo":true"#,
        r#", "images":["x.png"]"#, r#", "weights":"gguf""#, r#", "sampler":"euler""#,
        r#", "scheduler":"normal""#, r#", "clip_skip":0"#, r#", "steps":1"#, r#", "prompts":[]"#,
        r#", "negative_prompt":3"#, r#", "output_dir":"../escape""#,
    ] { assert!(prepare(&cfg, &request(extra)).is_err(), "{extra}"); }
    std::fs::remove_file(fixture).unwrap();
    assert!(prepare(&cfg, &request("")).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn live_catalog_add_remove_disable_and_defaults() {
    let root = std::env::temp_dir().join(format!("oaiy-live-media-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("weights.safetensors"), b"fixture").unwrap();
    std::fs::write(root.join("other.gguf"), b"fixture").unwrap();
    let write = |name: &str, j: Json| std::fs::write(root.join(name), j.to_json()).unwrap();
    write("controller.json", Json::obj([
        ("worker", Json::str("other.gguf")), ("controller_path", Json::str("other.gguf")),
        ("controller_name", Json::str("qwen")), ("controller_device", Json::Int(0)),
        ("image_device", Json::Int(1)), ("output_root", Json::str("outputs")),
    ]));
    let cfg = Config::read(&root).unwrap();
    let request = Json::obj([("prompt", Json::str("a fox"))]);
    let available = |kind: &str| cfg.capabilities().get(kind).unwrap().get("available").unwrap().as_bool().unwrap();
    assert!(!available("image") && !available("video"));
    assert!(prepare(&cfg, &request).is_err());
    write("image.json", Json::obj([
        ("base", Json::str(".")), ("transformer", Json::str("other.gguf")),
        ("safetensors_transformer", Json::str("weights.safetensors")),
        ("adapter", Json::str("weights.safetensors")), ("default_weights", Json::str("safetensors")),
    ]));
    assert!(available("image"));
    let queued = prepare(&cfg, &request).unwrap();
    assert!(queued.get("transformer").unwrap().as_str().unwrap().ends_with("weights.safetensors"));
    assert_eq!(queued.get("steps").and_then(Json::as_i64), Some(6));
    write("image.json", Json::obj([
        ("base", Json::str(".")), ("default_weights", Json::str("safetensors")),
        ("adapter", Json::str("weights.safetensors")), ("default_model", Json::str("red")),
        ("models", Json::obj([
            ("red", Json::obj([("safetensors_transformer", Json::str("weights.safetensors"))])),
            ("realism", Json::obj([("safetensors_transformer", Json::str("other.gguf")), ("text_encoder", Json::str("weights.safetensors"))])),
            ("off", Json::obj([("enabled", Json::Bool(false))])),
        ])),
    ]));
    let default = prepare(&cfg, &request).unwrap();
    assert_eq!(default.get("model").and_then(Json::as_str), Some("red"));
    let named = prepare(&cfg, &Json::obj([
        ("prompt",Json::str("fox")), ("model",Json::str("realism")),
        ("text_encoder",Json::str("untrusted")),
    ])).unwrap();
    assert!(named.get("transformer").unwrap().as_str().unwrap().ends_with("other.gguf"));
    assert!(named.get("text_encoder").unwrap().as_str().unwrap().ends_with("weights.safetensors"));
    assert_eq!(cfg.capabilities().get("image").unwrap().get("models").unwrap().as_array().unwrap().len(),3);
    for name in ["off", "missing", "../../weights.safetensors", ""] {
        assert!(prepare(&cfg,&Json::obj([("prompt",Json::str("fox")),("model",Json::str(name))])).is_err());
    }
    write("image.json", Json::obj([("enabled", Json::Bool(false))]));
    assert!(!available("image"));
    assert!(prepare(&cfg, &request).is_err());
    // Prepared jobs retain their snapshot even when the catalog changes.
    assert!(queued.get("transformer").unwrap().as_str().unwrap().ends_with("weights.safetensors"));
    let weights = Json::obj(["transformer", "text_encoder", "vae", "tokenizer"].map(|k| (k, Json::str("weights.safetensors"))));
    write("video.json", Json::obj([
        ("default_model", Json::str("sulphur-2")),
        ("models", Json::obj([("sulphur-2", weights)])),
    ]));
    assert!(available("video"));
    let video = prepare_video(&cfg, &request).unwrap();
    assert_eq!(video.get("model").and_then(Json::as_str), Some("sulphur-2"));
    assert!(Path::new(video.get("transformer").unwrap().as_str().unwrap()).is_absolute());
    std::fs::remove_file(root.join("video.json")).unwrap();
    assert!(!available("video"));
    assert!(prepare_video(&cfg, &request).is_err());
    std::fs::write(root.join("image.json"), b"invalid JSON").unwrap();
    assert!(!available("image"));
    assert!(Config::read(&root).is_ok()); // a broken optional manifest does not prevent chat startup
    std::fs::remove_dir_all(root).unwrap();
}
#[test]
fn a_video_catalogs_backend_reaches_the_worker_and_a_requests_does_not() {
    let root = std::env::temp_dir().join(format!("oaiy-video-backend-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let weight = root.join("fixture.safetensors");
    std::fs::write(&weight, b"fixture").unwrap();
    let manifest = root.join("video.json");
    let write = |catalog: Option<&str>, model: Option<&str>| {
        let mut fields: Vec<(&str, Json)> = ["transformer", "text_encoder", "vae", "tokenizer"].iter().map(|k| (*k, Json::str(weight.to_string_lossy()))).collect();
        if let Some(b) = model {
            fields.push(("backend", Json::str(b)));
        }
        let mut top = vec![("models", Json::obj([("ltx-2.3", Json::obj(fields))]))];
        if let Some(b) = catalog {
            top.push(("backend", Json::str(b)));
        }
        std::fs::write(&manifest, Json::obj(top).to_json()).unwrap();
    };
    let mut cfg = config();
    cfg.output_root = root.join("output");
    cfg.video_config = Some(manifest.clone());
    let body = |s: &str| Json::parse(s.as_bytes()).unwrap();
    let backend = |cfg: &Config, request: &str| prepare_video(cfg, &body(request)).map(|r| r.get("backend").and_then(Json::as_str).map(str::to_owned));
    write(None, None);
    assert_eq!(backend(&cfg, r#"{"model":"ltx-2.3","prompt":"x","backend":"webgpu"}"#).unwrap(), None, "a request's own is not taken");
    write(Some("webgpu"), None);
    assert_eq!(backend(&cfg, r#"{"model":"ltx-2.3","prompt":"x"}"#).unwrap().as_deref(), Some("webgpu"));
    write(Some("webgpu"), Some("cpu"));
    assert_eq!(backend(&cfg, r#"{"model":"ltx-2.3","prompt":"x"}"#).unwrap().as_deref(), Some("cpu"), "the model's over the catalog's");
    write(Some("cuda"), None);
    assert!(backend(&cfg, r#"{"model":"ltx-2.3","prompt":"x"}"#).unwrap_err().contains("no CUDA backend"), "a catalog written for the CUDA build says so");
    write(Some("vulkan"), None);
    assert!(backend(&cfg, r#"{"model":"ltx-2.3","prompt":"x"}"#).is_err(), "an unknown backend");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn video_requests_use_trusted_paths_and_bounded_memory() {
    let root = std::env::temp_dir().join(format!("oaiy-video-config-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let weight = root.join("fixture.safetensors");
    std::fs::write(&weight, b"fixture").unwrap();
    let model = Json::obj(
        ["transformer", "text_encoder", "vae", "tokenizer"]
            .map(|k| (k, Json::str(weight.to_string_lossy()))),
    );
    let manifest = root.join("video.json");
    std::fs::write(
        &manifest,
        Json::obj([
            ("models", Json::obj([("sulphur-2", model)])),
            ("ram_gb", Json::Int(4)),
            ("vram_gb", Json::Int(8)),
        ])
        .to_json(),
    )
    .unwrap();
    let mut cfg = config();
    cfg.output_root = root.join("output");
    cfg.video_config = Some(manifest);
    let valid = |extra: &str| {
        Json::parse(
            format!(r#"{{"model":"sulphur-2","prompt":"A bird in flight"{extra}}}"#).as_bytes(),
        )
        .unwrap()
    };
    let r = prepare_video(
        &cfg,
        &valid(r#", "memory":"ssd", "transformer":"untrusted", "device":99"#),
    )
    .unwrap();
    assert_eq!(r.get("kind").and_then(Json::as_str), Some("video"));
    assert_eq!(r.get("memory").and_then(Json::as_str), Some("ssd"));
    assert_eq!(r.get("device").and_then(Json::as_i64), Some(1));
    assert!(r.get("cache_dir").and_then(Json::as_str).unwrap().ends_with(".ltx-prompt-cache"));
    assert!(prepare_video(&cfg, &valid(r#", "prompt_cache":false"#)).unwrap().get("cache_dir").is_none());
    for key in ["image", "end_image"] {
        let reference = Json::obj([(key, Json::str(weight.to_string_lossy()))]);
        assert!(video_reference_path(&reference, key, false).is_err());
        let expected = video_reference_path(&reference, key, true).unwrap().unwrap();
        let mut body = valid("");
        if let Json::Obj(fields) = &mut body { fields.push((key.into(), Json::str(weight.to_string_lossy()))); }
        assert_eq!(prepare_video(&cfg, &body).unwrap().get(key).unwrap().to_json(), expected.to_json());
        assert!(video_reference_path(&Json::obj([(key, Json::Null)]), key, false).unwrap().is_none());
        for invalid in [Json::Int(1), Json::str("relative.png"), Json::str("")] {
            assert!(video_reference_path(&Json::obj([(key, invalid)]), key, true).is_err());
        }
    }
    assert_eq!(
        r.get("transformer").and_then(Json::as_str),
        Some(weight.to_str().unwrap())
    );
    for extra in [
        r#", "ram_gb":5"#,
        r#", "vram_gb":9"#,
        r#", "frames":48"#,
        r#", "steps":6"#,
        r#", "output_dir":"../outside""#,
        r#", "images":["x.png"]"#,
    ] {
        assert!(prepare_video(&cfg, &valid(extra)).is_err(), "{extra}");
    }
    std::fs::remove_file(weight).unwrap();
    assert!(prepare_video(&cfg, &valid(""))
        .unwrap_err()
        .contains("not ready"));
    std::fs::remove_dir_all(root).unwrap();
}
fn config() -> Config {
    Config {
        media_dir: None,
        default_weights: "gguf".into(),
        image_model: None,
        text_encoder: None,
        sdxl: None,
        klein: None,
        image_memory: Json::obj([] as [(&str, Json); 0]),
        worker: "worker".into(),
        base: "base".into(),
        transformer: "model.gguf".into(),
        safetensors_transformer: None,
        adapter: Some("turbo.safetensors".into()),
        loras: Vec::new(),
        output_root: std::env::temp_dir()
            .join(format!("oaiy-image-test-{}", std::process::id())),
        controller_name: "controller".into(),
        controller_path: "controller.gguf".into(),
        controller_device: 0,
        image_device: 1,
        video_config: None,
    }
}
#[test]
fn klein_catalog_dispatch_owns_paths_and_can_disable_style_loras() {
    let mut cfg=config();
    let root=std::env::temp_dir().join(format!("oaiy-klein-catalog-{}",std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    for file in ["transformer.safetensors","text.safetensors","vae.safetensors","tokenizer.json","style.safetensors"] { std::fs::write(root.join(file),b"fixture").unwrap(); }
    std::fs::write(root.join("image.json"),br#"{"default_model":"klein","models":{"klein":{"architecture":"flux2-klein-4b","transformer":"transformer.safetensors","text_encoder":"text.safetensors","vae":"vae.safetensors","tokenizer":"tokenizer.json","loras":[{"path":"style.safetensors","strength":0.75}]}}}"#).unwrap();
    cfg.media_dir=Some(root.clone());
    // (its batches' directories its own: the default root is another test's to remove)
    cfg.output_root=root.join("output");
    let request=prepare(&cfg,&Json::parse(br#"{"model":"klein","prompt":"fox","transformer":"untrusted","device":99}"#).unwrap()).unwrap();
    assert_eq!(request.get("architecture").and_then(Json::as_str),Some("flux2-klein-4b"));
    assert_eq!(request.get("transformer").and_then(Json::as_str),root.join("transformer.safetensors").to_str());
    assert_eq!(request.get("device").and_then(Json::as_i64),Some(1));
    assert_eq!(request.get("steps").and_then(Json::as_i64),Some(4));
    assert_eq!(request.get("cfg").and_then(Json::as_f64),Some(1.0));
    assert_eq!(request.get("loras").and_then(Json::as_array).unwrap().len(),1);
    let baseline=prepare(&cfg,&Json::parse(br#"{"model":"klein","prompt":"fox","use_loras":false}"#).unwrap()).unwrap();
    assert!(baseline.get("loras").and_then(Json::as_array).unwrap().is_empty());
    for negative in [Json::Null, Json::str(""), Json::str(" \t")] {
        let r=prepare(&cfg,&Json::obj([("model",Json::str("klein")),("prompt",Json::str("fox")),("negative_prompt",negative)])).unwrap();
        assert!(r.get("negative_prompt").is_none());
    }
    for negative in [Json::str("blur"), Json::Bool(false), Json::Arr(Vec::new())] {
        assert!(prepare(&cfg,&Json::obj([("model",Json::str("klein")),("prompt",Json::str("fox")),("negative_prompt",negative)])).is_err());
    }
    for bad in [br#"{"model":"klein","prompt":"fox","steps":8}"#.as_slice(),br#"{"model":"klein","prompt":"fox","cfg":4}"#.as_slice(),br#"{"model":"klein","prompt":"fox","input_reference":"untrusted"}"#.as_slice()] { assert!(prepare(&cfg,&Json::parse(bad).unwrap()).is_err()); }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_invalid_batches_before_creating_output() {
    let cfg = config();
    for json in [
        r#"{"prompt":"x","n":0}"#,
        r#"{"prompt":"x","n":1001}"#,
        r#"{"prompt":"x","steps":5}"#,
        r#"{"prompt":"x","width":257}"#,
        r#"{"prompt":"x","output_dir":"../escape"}"#,
        r#"{"prompts":["a","b"],"n":3}"#,
        r#"{"prompt":"x","weights":3}"#,
        r#"{"prompt":"x","images":["a","b","c","d"]}"#,
        r#"{"prompt":"x","images":"file.png"}"#,
        r#"{"prompt":"x","images":[null]}"#,
        r#"{"prompt":"x","images":["relative.png"]}"#,
        r#"{"prompt":"x","reference_size":1025}"#,
    ] {
        assert!(
            prepare(&cfg, &Json::parse(json.as_bytes()).unwrap()).is_err(),
            "{json}"
        );
    }
}
#[test]
fn optional_references_obey_local_file_policy_and_preserve_order() {
    assert!(reference_paths(&Json::obj([] as [(&str, Json); 0]), false)
        .unwrap()
        .is_empty());
    assert!(
        reference_paths(&Json::obj([("images", Json::Arr(vec![]))]), false)
            .unwrap()
            .is_empty()
    );
    let path = std::env::temp_dir().join(format!("oaiy-reference-{}.png", std::process::id()));
    std::fs::write(&path, b"fixture").unwrap();
    for count in 1..=3 {
        let body = Json::obj([(
            "images",
            Json::Arr(
                (0..count)
                    .map(|_| Json::str(path.to_string_lossy()))
                    .collect(),
            ),
        )]);
        assert!(reference_paths(&body, false).is_err());
        assert_eq!(reference_paths(&body, true).unwrap().len(), count);
    }
    std::fs::remove_file(path).unwrap();
}
#[test]
fn batch_parameters_and_server_owned_paths_are_preserved() {
    let mut cfg = config();
    let turbo = prepare(&cfg, &Json::parse(br#"{"prompt":"x"}"#).unwrap()).unwrap();
    assert_eq!(turbo.get("steps").and_then(Json::as_i64), Some(6));
    assert_eq!(
        turbo.get("adapter").and_then(Json::as_str),
        Some("turbo.safetensors")
    );
    let r=prepare(&cfg,&Json::parse(br#"{"prompt":"x","n":100,"steps":4,"weights":"safetensors","base":"untrusted","device":99}"#).unwrap()).unwrap();
    assert_eq!(r.get("n").and_then(Json::as_i64), Some(100));
    assert_eq!(r.get("steps").and_then(Json::as_i64), Some(4));
    assert_eq!(r.get("base").and_then(Json::as_str), Some("base"));
    assert_eq!(r.get("device").and_then(Json::as_i64), Some(1));
    assert_eq!(
        Path::new(r.get("transformer").unwrap().as_str().unwrap()),
        Path::new("base").join("transformer")
    );
    let base = prepare(
        &cfg,
        &Json::parse(br#"{"prompt":"x","turbo":false}"#).unwrap(),
    )
    .unwrap();
    assert_eq!(base.get("adapter"), Some(&Json::Null));
    assert_eq!(base.get("steps").and_then(Json::as_i64), Some(40));
    cfg.safetensors_transformer = Some("custom.safetensors".into());
    let custom = prepare(&cfg, &Json::parse(br#"{"prompt":"x","weights":"safetensors","transformer":"untrusted","safetensors_transformer":"untrusted"}"#).unwrap()).unwrap();
    assert_eq!(custom.get("transformer").and_then(Json::as_str), Some("custom.safetensors"));
    let gguf = prepare(&cfg, &Json::parse(br#"{"prompt":"x","weights":"gguf"}"#).unwrap()).unwrap();
    assert_eq!(gguf.get("transformer").and_then(Json::as_str), Some("model.gguf"));
    std::fs::remove_dir_all(&cfg.output_root).unwrap();
}
#[test]
fn a_catalogs_backend_reaches_the_worker_and_a_requests_does_not() {
    let mut cfg = config();
    cfg.output_root = std::env::temp_dir().join(format!("oaiy-image-backend-{}", std::process::id()));
    let body = |s: &str| Json::parse(s.as_bytes()).unwrap();
    let r = prepare(&cfg, &body(r#"{"prompt":"x","backend":"webgpu"}"#)).unwrap();
    assert!(r.get("backend").is_none(), "a request's own backend is not taken");
    cfg.image_memory = Json::obj([("backend", Json::str("webgpu"))]);
    let r = prepare(&cfg, &body(r#"{"prompt":"x"}"#)).unwrap();
    assert_eq!(r.get("backend").and_then(Json::as_str), Some("webgpu"));
    cfg.image_memory = Json::obj([("backend", Json::str("vulkan"))]);
    assert!(prepare(&cfg, &body(r#"{"prompt":"x"}"#)).is_err(), "an unknown backend");
}

#[test]
fn image_memory_defaults_to_auto_and_respects_catalog_caps() {
    let mut cfg = config();
    cfg.output_root = std::env::temp_dir().join(format!("oaiy-image-memory-{}", std::process::id()));
    let body = |s: &str| Json::parse(s.as_bytes()).unwrap();
    let r = prepare(&cfg, &body(r#"{"prompt":"x"}"#)).unwrap();
    assert_eq!(r.get("memory").and_then(Json::as_str), Some("auto"));
    assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(32));
    assert!(r.get("vram_gb").is_none());
    cfg.image_memory = body(r#"{"memory":"ssd","ram_gb":8,"vram_gb":12}"#);
    let r = prepare(&cfg, &body(r#"{"prompt":"x"}"#)).unwrap();
    assert_eq!(r.get("memory").and_then(Json::as_str), Some("ssd"));
    assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(8));
    assert_eq!(r.get("vram_gb").and_then(Json::as_i64), Some(12));
    let r = prepare(&cfg, &body(r#"{"prompt":"x","memory":"ram","ram_gb":4,"vram_gb":0}"#)).unwrap();
    assert_eq!(r.get("memory").and_then(Json::as_str), Some("ram"));
    assert_eq!(r.get("ram_gb").and_then(Json::as_i64), Some(4));
    assert_eq!(r.get("vram_gb").and_then(Json::as_i64), Some(0));
    for extra in [r#","ram_gb":9"#, r#","vram_gb":13"#, r#","memory":"disk""#, r#","ram_gb":"4""#] {
        assert!(prepare(&cfg, &body(&format!(r#"{{"prompt":"x"{extra}}}"#))).is_err(), "{extra}");
    }
    std::fs::remove_dir_all(&cfg.output_root).unwrap();
}
#[test]
fn configured_safetensors_override_is_optional_and_validated() {
    let root = std::env::temp_dir().join(format!("oaiy-safe-config-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let checkpoint = root.join("custom.safetensors");
    std::fs::write(&checkpoint, b"fixture").unwrap();
    let config_path = root.join("config.json");
    for (value, valid) in [
        (Json::Null, true),
        (Json::str(checkpoint.to_string_lossy()), true),
        (Json::Int(1), false),
        (Json::str(""), false),
        (Json::str("relative.safetensors"), false),
        (Json::str(root.join("missing.safetensors").to_string_lossy()), false),
    ] {
        let body = Json::obj([
            ("worker", Json::str(checkpoint.to_string_lossy())),
            ("base", Json::str(root.to_string_lossy())),
            ("transformer", Json::str(checkpoint.to_string_lossy())),
            ("safetensors_transformer", value),
            ("output_root", Json::str(root.to_string_lossy())),
            ("controller_name", Json::str("controller")),
            ("controller_path", Json::str(checkpoint.to_string_lossy())),
            ("controller_device", Json::Int(0)),
            ("image_device", Json::Int(1)),
        ]);
        std::fs::write(&config_path, body.to_json()).unwrap();
        assert_eq!(Config::read(&config_path).is_ok(), valid);
    }
    std::fs::remove_dir_all(root).unwrap();
}
