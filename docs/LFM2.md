# LFM2.5 24B-A2B

Liquid AI's LFM2.5-24B-A2B (October 2026, the LFM Open License 1.0: commercial use only below US$10M of annual
revenue) is a reasoning model for agents: 24B weights, 2.45B of them a token. It always thinks before it answers, and
it calls tools.

| GGUF (Liquid AI's) | Size | Architecture |
| --- | --- | --- |
| [`LFM2.5-24B-A2B-Q4_K_M.gguf`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) | 14.5 GB | `lfm2moe` |

**Get models → Chat → LFM2.5 24B-A2B** fetches it; or add the file yourself. In `oaiy-llm-server`:

```
oaiy-llm-server --model LFM2.5-24B-A2B-Q4_K_M.gguf --name lfm2.5
```

## The model

40 layers (transformers' `Lfm2MoeForCausalLM`), 2048 wide, a 128,000-token vocabulary, its embeddings its head:

- **30 gated short convs** (`Lfm2ShortConv`): the input's projection gives each row's B, C and x; `B * x` goes through
  a causal depthwise conv of three taps (the two products before a run its state), times C, then out.
- **10 attention layers**: 32 query heads and 8 key-value heads of 64, each head's q and k RMS-normed, RoPE (NeoX,
  base 10^7) over the whole head.
- **FFNs**: dense SwiGLUs (11,776 wide) in the first two layers; in the other 38, 64 experts (1,536 wide), 4 a token,
  no shared expert. Each expert's score is the sigmoid of its router logit. The top 4 are picked by score plus a
  selection bias (`exp_probs_b`), and weighted by their scores alone over the scores' sum.

## How OAIY runs it

As a Qwen3.5 hybrid (`llama_rs::Qwen35Model`), its short convs a recurrent layer of their own
(`Qwen35Block::Conv`): the server's Qwen engine serves it, with its prompt states, its chained runs on the GPU and its
tools.

- **The conv state** is the cache's conv window for the layer (`[2, 2048]`); the layer's delta-net state is a
  placeholder of one value, so what keeps a hybrid's states (a checkpoint, a disk state) keeps a conv layer's as it
  keeps a delta net's.
- **On the GPU** a conv layer is its in-projection, `ChainRecorder::short_conv` (B times x, the conv over its state,
  times C, in one kernel: a thread a channel for a step, a thread a channel and token for a prompt), and its
  out-projection. An attention layer is Qwen3.5's without the gate half of its q.
- **The experts** are held by the GPU's K-quant kernels (`ggml_rs_wgpu::quant_moe`, Q4_K and Q6_K). Their routing
  is `ggml_rs::exl3::Routing::Sigmoid`, on the host and on the GPU (`ROUTE_SIGMOID_PICK`, then the ranking Qwen's
  routing uses, then `ROUTE_SIGMOID_WEIGHTS`). A layer with no shared expert has that expert's place in a row weighted 0.
- **The tokenizer** is Llama 3's split with Hugging Face's `ignore_merges`: a piece that is a token whole is that
  token (`tokenizer.ggml.pre` `lfm2`, as llama.cpp reads it).

## Chat and tools

`crates/oaiy-llm-server/src/lfm2.rs` writes the prompt as its `chat_template.jinja` does: `<|startoftext|>`, ChatML,
the tools listed in the system message as JSON (`List of tools: [...]`), a reply's thinking kept after the last user
message and dropped before it, and the reply opened on `<think>`. A request that does not think gets `<think></think>`.

It writes calls as a Python list between `<|tool_call_start|>` and `<|tool_call_end|>`, for example
`[get_weather(city='Paris', units='celsius')]`. The server reads them into OpenAI's `tool_calls`. Strings in either
quote (with Python's escapes), numbers, booleans, `None`, lists and dicts are all read. JSON calls, where a system
prompt asks for them, are read too.

## Checks

- `lfm2::tests::lfm2s_engine_is_transformers`: a small LFM2-MoE of random weights (`tools/lfm2/reference.py`: dense
  conv and attention layers, then experts), its f32 GGUF made by llama.cpp's converter, against transformers 5.21 in
  float32. Each prompt's logits, then eight greedy steps through the cache: every row at a cosine of 1.0000000, the
  same tokens.
- `lfm2::tests::lfm2s_prompts_are_transformers`: seven conversations (tools with and without a system message, a call
  and its result, two calls whose strings need escapes, a history whose thinking is dropped) are transformers'
  `apply_chat_template`'s, character for character. Calls are read back as the template wrote them.
- The tokenizer (`tokenizer/tests/gguf_ids.rs`, the GGUF and Hugging Face's ids for eighteen texts in sixteen
  languages, code, numbers and whitespace runs): the same 513 ids.
- On the RTX 4090, `models::dense_webgpu_timing::a_chained_qwen35_run_answers_as_the_host_path` (QWEN35_MODEL the
  GGUF): the chained run and the host path agree on 600 prompt tokens' logits at a cosine of 0.9994. They part at the
  fourth step, at a near tie, as a MoE model's do.
- Against llama.cpp v0.7.0 (Metal) on three chats, greedy: the reply that calls a tool is the same for all 64
  tokens. One reply is the same for 18 tokens, then parts at a near tie. The third parts at its first token, where
  the two likeliest are within half a logit. llama.cpp's own CPU and Metal paths disagree by 0.3 logits there. OAIY's
  host path and chained run agree with each other.

## Speed

- **RTX 4090 (eGPU):** all 14.5 GB on the card; replies at about 220 tokens a second; 7,723 prompt tokens read in 1.4 s.
- **Apple M5 Pro (Metal):** all 14.5 GB on its GPU; replies at about 46 tokens a second.
