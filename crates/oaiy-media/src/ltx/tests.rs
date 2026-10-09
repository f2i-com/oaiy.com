use super::*;
#[test]
fn endpoint_tokens_stay_clean_without_freezing_the_last_video_latent() -> Result<()> {
    let start = Tensor::full(3f32, (1, 2, 2), &Device::Cpu)?;
    let end = Tensor::full(7f32, (1, 2, 2), &Device::Cpu)?;
    for with_start in [false, true] {
        let mut latent = Tensor::zeros((1, 8, 2), DType::F32, &Device::Cpu)?;
        for _ in 0..8 {
            latent = condition_endpoints(
                (latent + 0.25)?,
                with_start.then_some(&start),
                Some(&end),
            )?;
        }
        let values = latent.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(&values[..4], &[if with_start { 3. } else { 2. }; 4]);
        assert_eq!(&values[4..12], &[2.; 8]);
        assert_eq!(&values[12..], &[7.; 4]);
    }
    Ok(())
}
#[test]
fn soundtrack_level_matches_the_voice_and_never_clips() {
    let rate = 1000;
    // A loud tone (peak 1.4, RMS near 1) against a reference voice at RMS 0.4.
    let tone: Vec<f32> = (0..2 * rate).flat_map(|i| { let v = 1.4 * (i as f32 * 0.3).sin(); [v, v] }).collect();
    let mut matched = tone.clone();
    set_level(&mut matched, rate, Some(0.4));
    let left: Vec<f32> = matched.iter().step_by(2).copied().collect();
    assert!((speech_loudness(&left, rate).unwrap() - 0.4).abs() < 0.02);
    let mut limited = tone;
    set_level(&mut limited, rate, None);
    assert!(limited.iter().all(|x| x.abs() <= 0.967));
    assert_eq!(speech_loudness(&[0.; 100], rate), None);
}
#[test]
fn guided_schedule_matches_the_reference() {
    let s = guided_sigmas(30);
    assert_eq!(s.len(), 31);
    assert_eq!(s[0], 1.);
    assert!((s[29] - 0.1).abs() < 1e-9);
    assert_eq!(s[30], 0.);
    assert!(s.windows(2).all(|w| w[0] > w[1]));
    // Before stretching, t = 0.5 maps to e^2.05 / (e^2.05 + 1).
    let shifted = 2.05f64.exp() / (2.05f64.exp() + 1.);
    let last = 2.05f64.exp() / (2.05f64.exp() + 29.);
    let scale = (1. - last) / 0.9;
    assert!((s[15] - (1. - (1. - shifted) / scale)).abs() < 1e-12);
}
#[test]
fn clips_follow_the_soundtrack_length() {
    assert_eq!(frames_for_audio(3.2, 24), 81);
    // "After thirty years?": 1.68 s needs 40.3 frames, so 41, not 33.
    assert_eq!(frames_for_audio(1.68, 24), 41);
    assert_eq!(frames_for_audio(1.0, 24), 25);
    assert_eq!(frames_for_audio(0.2, 24), 9);
    assert_eq!(frames_for_audio(30., 24), 121);
    let base = |extra: &str| {
        // (the sound's request logic: WebGPU video refuses sound yet, so the backend these name is the CPU)
        Json::parse(format!(r#"{{"model":"ltx-2.5","transformer":"t","text_encoder":"e","vae":"v","audio_vae":"a","output_dir":"o","prompt":"p","backend":"cpu"{extra}}}"#).as_bytes()).unwrap()
    };
    let r = Request::parse(&base(r#","speech":{"kind":"speech"}"#)).unwrap();
    assert!(r.frames_from_audio && r.audio && r.speech.is_some());
    let r = Request::parse(&base(r#","speech":{"kind":"speech"},"frames":49"#)).unwrap();
    assert!(!r.frames_from_audio && r.frames == 49);
    assert!(Request::parse(&base(r#","speech":{"kind":"speech"},"audio":false"#)).is_err());
    assert!(Request::parse(&base(r#","audio_file":"relative.wav""#)).is_err());
}
#[test]
fn audio_length_follows_the_clip() {
    assert_eq!(audio_latent_frames(49, 24), 51);
    assert_eq!(audio_latent_frames(121, 24), 126);
    assert_eq!(audio_latent_frames(25, 25), 25);
}
#[test]
fn audio_defaults_on_for_ltx_2_5_with_an_audio_vae() {
    let base = |extra: &str| {
        Json::parse(format!(r#"{{"model":"ltx-2.5","transformer":"t","text_encoder":"e","vae":"v","output_dir":"o","prompt":"p","backend":"cpu"{extra}}}"#).as_bytes()).unwrap()
    };
    assert!(Request::parse(&base(r#","audio_vae":"a""#)).unwrap().audio);
    let sulphur = Json::parse(br#"{"model":"sulphur-2","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","audio_vae":"t","output_dir":"o","prompt":"p","backend":"cpu"}"#).unwrap();
    assert!(Request::parse(&sulphur).unwrap().audio, "LTX 2.3 checkpoints carry their own audio VAE");
    assert!(!Request::parse(&base("")).unwrap().audio);
    assert!(!Request::parse(&base(r#","audio_vae":"a","audio":false"#)).unwrap().audio);
    assert!(Request::parse(&base(r#","audio":true"#)).is_err(), "audio without its VAE");
}
#[test]
#[cfg(feature = "webgpu")]
fn a_webgpu_clip_has_its_own_sound_or_follows_a_given_one() {
    let request = |extra: &str| {
        let j = Json::parse(format!(r#"{{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","audio_vae":"t","output_dir":"o","prompt":"p","backend":"webgpu"{extra}}}"#).as_bytes()).unwrap();
        Request::parse(&j)
    };
    let r = request("").unwrap();
    assert!(r.webgpu && r.audio, "a clip's own sound where the checkpoint has its audio VAE, on WebGPU as elsewhere");
    assert!(!request(r#","audio":false"#).unwrap().audio, "a silent clip where the job says so");
    assert!(request(r#","guidance":{"steps":30}"#).unwrap().guidance.is_some(), "guided sampling, its negative prompt by CFG");
    assert!(request(r#","negative_prompt":"n""#).unwrap().negative_prompt.is_some(), "a negative prompt without CFG: by NAG");
    assert!(request(r#","lora":"l""#).unwrap().lora.is_some(), "a LoRA: the weights it adapts loaded with it");
    let spoken = request(r#","speech":{"text":"hello"}"#).unwrap();
    assert!(spoken.speech.is_some() && spoken.audio && spoken.frames_from_audio, "speech made first and followed, the clip as long as it");
    for (extra, what) in [
        (r#","identity":true"#, "reference voice"),
        (r#","refine":{"transformer":"t","upsampler":"u"}"#, "refinement"),
    ] {
        let e = request(extra).unwrap_err();
        assert!(e.starts_with("WebGPU video does not support") && e.contains(what), "{extra}: {e}");
    }
    let cpu = Json::parse(br#"{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","output_dir":"o","prompt":"p","backend":"cpu"}"#).unwrap();
    assert!(!Request::parse(&cpu).unwrap().webgpu);
    let cuda = Json::parse(br#"{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","output_dir":"o","prompt":"p","backend":"cuda"}"#).unwrap();
    assert!(Request::parse(&cuda).unwrap_err().contains("no CUDA backend"));
    let other = Json::parse(br#"{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","output_dir":"o","prompt":"p","backend":"metal"}"#).unwrap();
    assert!(Request::parse(&other).is_err());
}
#[test]
fn starting_frame_is_exact_after_each_euler_update() -> Result<()> {
    let clean = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &Device::Cpu)?;
    let mut latent = Tensor::zeros((1, 6, 2), DType::F32, &Device::Cpu)?;
    for _ in 0..8 {
        latent = condition_endpoints((latent + 0.25)?, Some(&clean), None)?;
        assert_eq!(
            latent.narrow(1, 0, 2)?.flatten_all()?.to_vec1::<f32>()?,
            vec![1., 2., 3., 4.]
        );
    }
    assert_eq!(
        latent.narrow(1, 2, 4)?.flatten_all()?.to_vec1::<f32>()?,
        vec![2.; 8]
    );
    Ok(())
}
