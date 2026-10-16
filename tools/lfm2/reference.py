"""tools/lfm2/reference.py: transformers' LFM2-MoE (`Lfm2MoeForCausalLM`) on a small model of random weights, the
reference OAIY's engine is held against (`lfm2::tests::lfm2s_engine_is_transformers`): every piece of the real
model's (gated short convs, GQA attention with per-head q and k norms, dense FFNs in its first layers, sigmoid-routed
experts with a selection bias, tied embeddings), at a size its f32 GGUF is 70 MB.

    python3 tools/lfm2/reference.py TOKENIZER_DIR OUT_DIR
    python3 LLAMA_CPP/convert_hf_to_gguf.py OUT_DIR/hf --outtype f32 --outfile OUT_DIR/lfm2-tiny-f32.gguf

TOKENIZER_DIR holds an LFM2.5 tokenizer (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`: LFM2.5-24B-
A2B's). OUT_DIR/hf gets the model in the Hub's layout (each expert's w1, w2 and w3, as LiquidAI's checkpoints and
llama.cpp's converter have them) and OUT_DIR/reference.json each prompt's ids, the logits after it (float32, on the
CPU) and greedy steps after it, each step's token and the logits after it (through transformers' cache: the conv
states and the attention's keys and values as a reply's steps use them). Needs torch, transformers (5.21: LFM2-MoE's
experts fused as it loads them) and safetensors.
"""
import json, os, shutil, sys
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Lfm2MoeForCausalLM

tokdir, out = sys.argv[1], sys.argv[2]
hf = os.path.join(out, "hf")
os.makedirs(hf, exist_ok=True)
h, ff, moe_ff, heads, kv_heads, experts, top_k, vocab = 128, 256, 64, 4, 2, 8, 2, 128000
hd = h // heads
# a dense conv layer, a dense attention layer, then experts: conv, conv, attention, conv
types = ["conv", "full_attention", "conv", "conv", "full_attention", "conv"]
config = {
    "architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe", "dtype": "float32",
    "hidden_size": h, "intermediate_size": ff, "moe_intermediate_size": moe_ff, "num_hidden_layers": len(types), "layer_types": types,
    "num_attention_heads": heads, "num_key_value_heads": kv_heads, "num_experts": experts, "num_experts_per_tok": top_k, "num_dense_layers": 2,
    "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": False, "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    "max_position_embeddings": 4096, "tie_word_embeddings": True, "use_expert_bias": True, "norm_topk_prob": True, "routed_scaling_factor": 1.0,
    "vocab_size": vocab, "bos_token_id": 124894, "eos_token_id": 124900, "pad_token_id": 124893, "initializer_range": 0.02, "use_cache": True,
}
json.dump(config, open(os.path.join(hf, "config.json"), "w"), indent=1)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    shutil.copy(os.path.join(tokdir, f), hf)

g = torch.Generator().manual_seed(20261016)
def rnd(*shape, scale=1.0):
    return (torch.randn(*shape, generator=g) * scale).contiguous()
def norm(n):
    return (1.0 + rnd(n, scale=0.1)).contiguous()
sd = {"model.embed_tokens.weight": rnd(vocab, h, scale=0.5), "model.embedding_norm.weight": norm(h)}
for i, t in enumerate(types):
    p = f"model.layers.{i}."
    sd[p + "operator_norm.weight"] = norm(h)
    sd[p + "ffn_norm.weight"] = norm(h)
    if t == "conv":
        sd[p + "conv.conv.weight"] = rnd(h, 1, 3, scale=0.5)
        sd[p + "conv.in_proj.weight"] = rnd(3 * h, h, scale=h ** -0.5)
        sd[p + "conv.out_proj.weight"] = rnd(h, h, scale=h ** -0.5)
    else:
        sd[p + "self_attn.q_proj.weight"] = rnd(heads * hd, h, scale=h ** -0.5)
        sd[p + "self_attn.k_proj.weight"] = rnd(kv_heads * hd, h, scale=h ** -0.5)
        sd[p + "self_attn.v_proj.weight"] = rnd(kv_heads * hd, h, scale=h ** -0.5)
        sd[p + "self_attn.out_proj.weight"] = rnd(h, heads * hd, scale=(heads * hd) ** -0.5)
        sd[p + "self_attn.q_layernorm.weight"] = norm(hd)
        sd[p + "self_attn.k_layernorm.weight"] = norm(hd)
    if i < config["num_dense_layers"]:
        sd[p + "feed_forward.w1.weight"] = rnd(ff, h, scale=h ** -0.5)
        sd[p + "feed_forward.w3.weight"] = rnd(ff, h, scale=h ** -0.5)
        sd[p + "feed_forward.w2.weight"] = rnd(h, ff, scale=ff ** -0.5)
    else:
        sd[p + "feed_forward.gate.weight"] = rnd(experts, h, scale=h ** -0.5)
        # a bias large enough to change which experts some rows take
        sd[p + "feed_forward.expert_bias"] = rnd(experts, scale=0.2)
        for e in range(experts):
            sd[p + f"feed_forward.experts.{e}.w1.weight"] = rnd(moe_ff, h, scale=h ** -0.5)
            sd[p + f"feed_forward.experts.{e}.w3.weight"] = rnd(moe_ff, h, scale=h ** -0.5)
            sd[p + f"feed_forward.experts.{e}.w2.weight"] = rnd(h, moe_ff, scale=moe_ff ** -0.5)
save_file(sd, os.path.join(hf, "model.safetensors"), metadata={"format": "pt"})

model = Lfm2MoeForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(hf)
prompts = [
    tok.apply_chat_template([{"role": "user", "content": "What is C. elegans? Answer in two sentences."}], tokenize=False, add_generation_prompt=True),
    "The quick brown fox jumps over the lazy dog. " * 12,
]
cases = []
with torch.no_grad():
    for text in prompts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        o = model(torch.tensor([ids]), use_cache=True)
        logits, past = o.logits[0, -1], o.past_key_values
        steps = []
        for _ in range(8):
            nxt = int(torch.argmax(logits))
            o = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            logits, past = o.logits[0, -1], o.past_key_values
            steps.append({"token": nxt, "logits": logits.tolist()})
        cases.append({"ids": ids, "logits": model(torch.tensor([ids])).logits[0, -1].tolist(), "steps": steps})
        print(len(ids), "prompt tokens; greedy:", [s["token"] for s in steps])
json.dump({"vocab": vocab, "cases": cases}, open(os.path.join(out, "reference.json"), "w"))
