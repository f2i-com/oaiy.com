use super::*;

fn thinking() -> Options {
    Options { mode: Mode::Thinking, effort: 75, drop_thinking: true }
}

/// The request's messages as the server hands them on: the tools on the first message, a system message (empty) put
/// first where there is none (`api.rs`).
fn as_served(messages: &Json, tools: &Json) -> Vec<Json> {
    let mut msgs = messages.as_array().unwrap().to_vec();
    if let Some(t) = tools.as_array() {
        if msgs[0].get("role").and_then(Json::as_str) != Some("system") {
            msgs.insert(0, Json::obj([("role", Json::str("system")), ("content", Json::str(""))]));
        }
        if let Json::Obj(first) = &mut msgs[0] {
            first.push(("tools".into(), Json::Arr(t.to_vec())));
        }
    }
    msgs
}

/// A prompt is transformers' (`apply_chat_template` of LFM2.5-24B-A2B's `chat_template.jinja`, the generation prompt
/// added; `template_cases.json`): plain, with a system message, a history whose earlier thinking is dropped, tools with
/// and without a system message, a call (its thinking kept, after the last user message) and its result, and two calls
/// whose strings need escapes. Each with the calls' arguments as objects, and as the JSON text OpenAI's clients send.
#[test]
fn lfm2s_prompts_are_transformers() {
    let cases = Json::parse(include_bytes!("template_cases.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case.get("name").and_then(Json::as_str).unwrap();
        let want = case.get("prompt").and_then(Json::as_str).unwrap();
        let msgs = as_served(case.get("messages").unwrap(), case.get("tools").unwrap());
        let got = chat_prompt(&msgs, &thinking()).unwrap().prompt;
        assert_eq!(got, want, "{name}");
        // the calls' arguments as JSON text
        let texted: Vec<Json> = msgs
            .iter()
            .map(|m| {
                let Json::Obj(fields) = m else { return m.clone() };
                Json::Obj(fields.iter().map(|(k, v)| match (k.as_str(), v) {
                    ("tool_calls", Json::Arr(calls)) => (k.clone(), Json::Arr(calls.iter().map(|c| {
                        let Json::Obj(cf) = c else { return c.clone() };
                        Json::Obj(cf.iter().map(|(ck, cv)| match (ck.as_str(), cv) {
                            ("function", Json::Obj(f)) => (ck.clone(), Json::Obj(f.iter().map(|(fk, fv)| if fk == "arguments" { (fk.clone(), Json::str(fv.to_json())) } else { (fk.clone(), fv.clone()) }).collect())),
                            _ => (ck.clone(), cv.clone()),
                        }).collect())
                    }).collect())),
                    _ => (k.clone(), v.clone()),
                }).collect())
            })
            .collect();
        assert_eq!(chat_prompt(&texted, &thinking()).unwrap().prompt, want, "{name}, arguments as JSON text");
    }
}

/// A request that does not think has its thought closed at once.
#[test]
fn a_reply_that_does_not_think_starts_after_an_empty_thought() {
    let msgs = Json::parse(br#"[{"role":"user","content":"Hi"}]"#).unwrap();
    let p = chat_prompt(msgs.as_array().unwrap(), &Options::default()).unwrap().prompt;
    assert_eq!(p, "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>");
}

fn tools() -> Vec<Json> {
    let cases = Json::parse(include_bytes!("template_cases.json")).unwrap();
    cases.as_array().unwrap().iter().find(|c| c.get("name").and_then(Json::as_str) == Some("two-calls")).unwrap().get("tools").unwrap().as_array().unwrap().to_vec()
}

fn dsml(calls: &[(&str, &[(&str, bool, &str)])]) -> String {
    let mut out = format!("<{DSML} calls>");
    for (name, args) in calls {
        out.push_str(&format!("<{DSML} invoke name=\"{name}\">"));
        for (k, s, raw) in *args {
            out.push_str(&format!("<{DSML} parameter name=\"{k}\" string=\"{s}\">{raw}</{DSML} parameter>"));
        }
        out.push_str(&format!("</{DSML} invoke>"));
    }
    out + &format!("</{DSML} calls>")
}

/// LFM2's calls (a Python list between its tags, as its card's example writes them, and as the template writes a
/// history's) are the server's tool markup: strings in either quote with their escapes, numbers, booleans, lists and
/// dicts; the text around them before; several calls, several blocks; a number where the parameter is a string, a
/// string; JSON calls (where a system prompt asks for them) as well.
#[test]
fn lfm2s_calls_are_the_servers_markup() {
    let tools = tools();
    let n = |t: &str| normalize(t, &tools).unwrap();
    assert_eq!(n("Plain answer."), "Plain answer.");
    assert_eq!(
        n("I'll look.</think><|tool_call_start|>[get_weather(city=\"Paris\", days=3)]<|tool_call_end|>Checking the weather."),
        format!("I'll look.</think>Checking the weather.{}", dsml(&[("get_weather", &[("city", true, "Paris"), ("days", false, "3")])]))
    );
    assert_eq!(
        n("<|tool_call_start|>[search(query='O\\'Brien\\'s \"notes\"\\nback\\\\slash', filters={'year': 2026, 'tags': ['a', 'b'], 'x': None}, exact=True), get_weather(city='Zürich')]<|tool_call_end|>"),
        dsml(&[("search", &[("query", true, "O'Brien's \"notes\"\nback\\slash"), ("filters", false, r#"{"year":2026,"tags":["a","b"],"x":null}"#), ("exact", false, "true")]), ("get_weather", &[("city", true, "Zürich")])])
    );
    // a number where the parameter is a string; two blocks; no list brackets
    assert_eq!(
        n("A<|tool_call_start|>get_weather(city=12345, units='celsius')<|tool_call_end|>B<|tool_call_start|>[search(query=\"x\", exact=False,)]<|tool_call_end|>"),
        format!("AB{}", dsml(&[("get_weather", &[("city", true, "12345"), ("units", true, "celsius")]), ("search", &[("query", true, "x"), ("exact", false, "false")])]))
    );
    // JSON calls
    assert_eq!(
        n(r#"<|tool_call_start|>[{"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}]<|tool_call_end|>"#),
        dsml(&[("get_weather", &[("city", true, "Oslo"), ("days", false, "2")])])
    );
    // refused: cut short, an undeclared function, a malformed call, a duplicate parameter, an end with no start
    for bad in [
        "<|tool_call_start|>[get_weather(city='Paris')",
        "<|tool_call_start|>[launch(code=1)]<|tool_call_end|>",
        "<|tool_call_start|>[get_weather(city=Paris)]<|tool_call_end|>",
        "<|tool_call_start|>[get_weather(city='a', city='b')]<|tool_call_end|>",
        "done<|tool_call_end|>",
    ] {
        assert!(normalize(bad, &tools).is_err(), "{bad:?}");
    }
}

/// A call as the template writes it reads back as the arguments it was written from.
#[test]
fn a_call_written_by_the_template_reads_back() {
    let tools = tools();
    let call = Json::parse(br#"[{"function":{"name":"search","arguments":{"query":"tab\there 'q' \"dq\" \\ end\n","filters":{"k":[1,2.5,true,null]},"exact":false}}}]"#).unwrap();
    let written = render_calls(call.as_array().unwrap()).unwrap();
    assert_eq!(
        normalize(&written, &tools).unwrap(),
        dsml(&[("search", &[("query", true, "tab\there 'q' \"dq\" \\ end\n"), ("filters", false, r#"{"k":[1,2.5,true,null]}"#), ("exact", false, "false")])])
    );
}

/// The cosine of two rows of logits.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

/// LFM2-MoE in OAIY's engine (a Qwen3.5 hybrid of short convs) is transformers' `Lfm2MoeForCausalLM`: a small model of
/// random weights (`tools/lfm2/reference.py`: two short convs and an attention layer dense, then experts; its f32 GGUF
/// made by llama.cpp's converter: OAIY_LFM2_GGUF, its reference OAIY_LFM2_REFERENCE), op by op on the CPU, its experts
/// the host's (`quant_experts_cpu`): each prompt's logits, then eight greedy steps' through the cache (the conv states
/// and the keys and values carried), every row's at a cosine past 0.99999 and its largest logit transformers'.
#[cfg(feature = "webgpu")]
#[test]
#[ignore = "needs OAIY_LFM2_GGUF and OAIY_LFM2_REFERENCE (tools/lfm2/reference.py)"]
fn lfm2s_engine_is_transformers() {
    use std::sync::Arc;
    let (Ok(path), Ok(reference)) = (std::env::var("OAIY_LFM2_GGUF"), std::env::var("OAIY_LFM2_REFERENCE")) else { return };
    let backend: Arc<dyn ggml_rs::Backend> = Arc::new(ggml_rs::CpuBackend::new());
    let g = gguf::GgufFile::open(&path).unwrap();
    let make = |_: usize, e: crate::llm::ExpertBlocks| -> crate::llm::Result<Box<dyn ggml_rs::exl3::Experts>> {
        let data = ggml_rs_wgpu::quant_moe::QuantExpertsData { hidden: e.hidden, ff: e.ff, experts: e.experts, gate: e.gate, up: e.up, down: e.down, shared: e.shared, routing: e.routing };
        ggml_rs_wgpu::quant_moe::quant_experts_cpu(data).map_err(crate::llm::LlamaError::Config)
    };
    let m = crate::llm::Qwen35Model::from_gguf_with(&g, Arc::clone(&backend), Some(&make)).unwrap();
    assert!(m.blocks.iter().any(|b| matches!(b, crate::llm::Qwen35Block::Conv { moe: Some(_), .. })) && m.blocks.iter().any(|b| matches!(b, crate::llm::Qwen35Block::Attention { moe: None, .. })));
    let r = Json::parse(&std::fs::read(reference).unwrap()).unwrap();
    let floats = |v: &Json| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect() };
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
    let mut worst = 1.0f64;
    for (n, case) in r.get("cases").unwrap().as_array().unwrap().iter().enumerate() {
        let ids: Vec<u32> = case.get("ids").unwrap().as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as u32).collect();
        let cfg = &m.config;
        let mut kv = crate::llm::KvCache::new(&*backend, cfg.n_layers, ids.len() + 16, cfg.n_kv_heads, cfg.head_dim);
        let row = |tokens: &[u32], kv: &mut crate::llm::KvCache| m.forward_embeds_host(&m.embed_text(tokens), tokens.len(), kv, None).unwrap().to_host().data().to_vec();
        let got = row(&ids, &mut kv);
        let want = floats(case.get("logits").unwrap());
        let c = cosine(&got, &want);
        eprintln!("prompt {n} ({} tokens): cosine {c:.7}, its token {} (transformers' {})", ids.len(), argmax(&got), argmax(&want));
        assert_eq!(argmax(&got), argmax(&want));
        worst = worst.min(c);
        for (i, step) in case.get("steps").unwrap().as_array().unwrap().iter().enumerate() {
            let token = step.get("token").unwrap().as_i64().unwrap() as u32;
            let got = row(&[token], &mut kv);
            let want = floats(step.get("logits").unwrap());
            let c = cosine(&got, &want);
            assert_eq!(argmax(&got), argmax(&want), "prompt {n}, step {i}");
            worst = worst.min(c);
        }
    }
    eprintln!("every row's logits at a cosine of {worst:.7} at least");
    assert!(worst > 0.99999, "{worst}");
}
