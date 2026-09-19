"""DeepSeek-V4.1 numerical oracle: the reference `inference/model.py`, run unmodified on
one GPU by streaming what does not fit.

- `kernel` (tilelang) is replaced by `torch_kernels` (pure torch, same arithmetic).
- Routed experts (289 GB) are `LazyExpert`s: read from the safetensors shards on first
  use and kept in a byte-budgeted GPU LRU.
- Engram tables (203 GB) are `LazyEngramEmbedding`s: row lookups through a memmap.
- Everything else (~13.5 GB with the reference's bf16/fp32 promotions) is resident.

Outputs (in --out):
  golden.safetensors       prompt ids, engram hash ids, embedding, every layer's output
                           and pre_mix after prefill, final logits, generated ids, the
                           first decode step's layer outputs, and one expert fixture
  engram_meta.safetensors  token_map / primes / offsets / multipliers / pad_id — the
                           Unicode-normalization + sympy + numpy-RNG precompute the Rust
                           runtime loads instead of recomputing
  <golden>.run.json        prompt text, decoded completion, timings, expert-cache stats,
                           and any test-only config overrides

Usage: python oracle.py --prompt "What is the capital of France?" --new-tokens 16
       python oracle.py --prompt-file prompts/long.txt --golden-name golden_long.safetensors \
                        --index-topk 32 --candidate-topk-blocks 4 --new-tokens 8
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch_kernels  # noqa: E402

sys.modules["kernel"] = torch_kernels  # must precede the reference import

REF = os.environ.get("DSV41_REF", r"E:\deepseek\reference")
sys.path.insert(0, os.path.join(REF, "inference"))
sys.path.insert(0, os.path.join(REF, "encoding"))
import model as M  # noqa: E402  (reference inference/model.py)
from encoding import encode_messages  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from st_index import StIndex  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402


class ExpertCache:
    """Routed expert weights on the GPU, LRU under a byte budget, loaded from the shards."""

    def __init__(self, idx: StIndex, device: str, budget_bytes: int):
        self.idx, self.device, self.budget = idx, device, budget_bytes
        self.lru: OrderedDict = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        self.load_seconds = 0.0

    def get(self, layer: int, expert: int):
        key = (layer, expert)
        if key in self.lru:
            self.hits += 1
            self.lru.move_to_end(key)
            return self.lru[key]
        self.misses += 1
        t0 = time.perf_counter()
        parts, nbytes = {}, 0
        for w in ("w1", "w2", "w3"):
            base = f"layers.{layer}.ffn.experts.{expert}.{w}"
            weight = self.idx.get(base + ".weight", self.device).view(torch.float4_e2m1fn_x2)
            weight.scale = self.idx.get(base + ".scale", self.device)
            parts[w] = weight
            nbytes += weight.numel() + weight.scale.numel()
        self.load_seconds += time.perf_counter() - t0
        self.lru[key] = parts
        self.bytes += nbytes
        while self.bytes > self.budget and len(self.lru) > 1:
            _, old = self.lru.popitem(last=False)
            self.bytes -= sum(p.numel() + p.scale.numel() for p in old.values())
        return parts


CACHE: ExpertCache | None = None
FIXTURE = {"layer": -1, "records": []}  # expert fixture capture


class LazyExpert(nn.Module):
    """Stands in for a routed `Expert`; forward is `Expert.forward` verbatim, weights fetched."""

    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.layer = self.expert = -1

    def forward(self, x, weights=None):
        p = CACHE.get(self.layer, self.expert)
        x_in, dtype = x, x.dtype
        gate = M.linear(x, p["w1"]).float()
        up = M.linear(x, p["w3"]).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = torch.nn.functional.silu(gate) * up
        if weights is not None:
            x = weights * x
        y = M.linear(x.to(dtype), p["w2"])
        if self.layer == FIXTURE["layer"] and len(FIXTURE["records"]) < 2:
            FIXTURE["records"].append((self.expert, x_in.detach().cpu(), None if weights is None else weights.detach().cpu(), y.detach().cpu()))
        return y


_OrigExpert = M.Expert


def expert_factory(dim, inter_dim, dtype=None, swiglu_limit=0.0):
    if dtype == torch.float4_e2m1fn_x2:  # routed experts; the shared expert stays real
        return LazyExpert(swiglu_limit)
    return _OrigExpert(dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)


class LazyEngramEmbedding(nn.Module):
    """`ParallelEngramEmbedding.forward` (world_size 1) over memmapped table rows."""

    IDX: StIndex | None = None

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, 32
        self.layer = -1
        self._w = self._s = None

    def forward(self, indices):
        if self._w is None:
            self._w = self.IDX.memmap_rows(f"layers.{self.layer}.engram.embed.weight")
            self._s = self.IDX.memmap_rows(f"layers.{self.layer}.engram.embed.scale")
        flat = indices.reshape(-1).cpu().numpy()
        assert flat.min() >= 0 and flat.max() < self.num_embeddings
        dev = indices.device
        vals = torch.from_numpy(np.ascontiguousarray(self._w[flat])).to(dev).view(torch.float8_e4m3fn).float()
        scales = torch_kernels.e8m0_to_f32(torch.from_numpy(np.ascontiguousarray(self._s[flat])).to(dev))
        values = (vals.unflatten(-1, (-1, self.block_size)) * scales.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
        return values.reshape(*indices.shape, self.dim)


def load_trunk(model: nn.Module, idx: StIndex):
    """Copy every resident parameter from the checkpoint; report anything unmatched."""
    loaded, missing = 0, []
    for name, t in model.state_dict().items():
        if name not in idx:
            missing.append(name)
            continue
        if name.endswith("attn.wo_a.weight"):  # convert.py: fp8 32x32 blocks -> bf16
            src = torch_kernels.dequant_fp8_weight(idx.get(name, t.device), idx.get(name[:-6] + "scale", t.device))
        else:
            src = idx.get(name, t.device)
        assert tuple(src.shape) == tuple(t.shape), (name, src.shape, t.shape)
        if t.element_size() == 1 and src.dtype == t.dtype:
            t.view(torch.uint8).copy_(src.view(torch.uint8))
        else:
            t.copy_(src.to(t.dtype))
        loaded += t.numel() * t.element_size()
    if missing:
        raise RuntimeError(f"{len(missing)} model tensors not in checkpoint, e.g. {missing[:5]}")
    return loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"E:\deepseek\model")
    ap.add_argument("--out", default=r"E:\deepseek\golden")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--device", type=int, default=1, help="CUDA device (1 = the x4-linked card here)")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--expert-cache-gb", type=float, default=14.0)
    ap.add_argument("--fixture-layer", type=int, default=3)
    ap.add_argument("--prompt-file", help="read the user message from a UTF-8 file instead of --prompt")
    ap.add_argument("--golden-name", default="golden.safetensors")
    # Test-only overrides. The real limits (top 512 positions, 2048 candidate
    # blocks) only bite past ~1k tokens, which a CPU reference cannot prefill in
    # reasonable time; shrinking them makes a ~200-token prompt exercise the
    # top-k selection and candidate filtering. Record them in run.json.
    ap.add_argument("--index-topk", type=int)
    ap.add_argument("--candidate-topk-blocks", type=int)
    args_cli = ap.parse_args()
    if args_cli.prompt_file:
        with open(args_cli.prompt_file, encoding="utf-8") as f:
            args_cli.prompt = f.read().strip()
    os.makedirs(args_cli.out, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dev = f"cuda:{args_cli.device}"
    torch.cuda.set_device(args_cli.device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(dev)
    torch.manual_seed(0)

    t_start = time.perf_counter()
    idx = StIndex(args_cli.model)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args_cli.model)
    with open(os.path.join(REF, "inference", "config.json")) as f:
        args = M.ModelArgs(**json.load(f))
    args.max_batch_size, args.max_seq_len, args.temperature = 1, args_cli.max_seq_len, 0.0
    args.vision_n_layers = 0  # text only: no ViT, no VL routing bias
    args.dspark_block_size = 0  # no MTP / DSpark draft layers
    overrides = {}
    if args_cli.index_topk is not None:
        args.index_topk = overrides["index_topk"] = args_cli.index_topk
    if args_cli.candidate_topk_blocks is not None:
        args.candidate_topk_blocks = overrides["candidate_topk_blocks"] = args_cli.candidate_topk_blocks

    # Same deliberate deviation as crates/dsv41/src/attention.rs: the reference
    # `Indexer.forward` publishes `shared_attn.index_k` only when a new latent
    # arrived, so on a ratio-2 decode step that completes no group, owning
    # indexers (layers 2/8/14) score against layer 20's stale ratio-1 keys. An
    # owner's own cache is always the right one; publish it unconditionally.
    _orig_indexer_forward = M.Indexer.forward

    def indexer_forward(self, x, qr, latent, start_pos, offset):
        if self.owns_k:
            M.shared_attn.index_k = self.k_cache
        return _orig_indexer_forward(self, x, qr, latent, start_pos, offset)

    M.Indexer.forward = indexer_forward

    global CACHE
    CACHE = ExpertCache(idx, dev, int(args_cli.expert_cache_gb * 2**30))
    FIXTURE["layer"] = args_cli.fixture_layer
    LazyEngramEmbedding.IDX = idx
    M.Expert = expert_factory
    M.ParallelEngramEmbedding = LazyEngramEmbedding

    t0 = time.perf_counter()
    model = M.Transformer(args, tokenizer)  # builds the engram token map (slow-ish)
    for i, layer in enumerate(model.layers):
        for e, ex in enumerate(layer.ffn.experts):
            ex.layer, ex.expert = i, e
        if layer.engram is not None:
            layer.engram.embed.layer = i
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    trunk_bytes = load_trunk(model, idx)
    t_load = time.perf_counter() - t0
    print(f"built in {t_build:.1f}s, trunk {trunk_bytes / 1e9:.2f} GB loaded in {t_load:.1f}s", flush=True)

    # engram precompute for the Rust runtime
    eh = model.engram_hash
    save_file(
        {
            "token_map": eh.token_map.cpu().int().contiguous(),
            "primes": eh.primes.cpu().long().contiguous(),
            "offsets": eh.offsets.cpu().long().contiguous(),
            "multipliers": eh.multipliers.cpu().long().contiguous(),
            "pad_id": torch.tensor([eh.pad_id], dtype=torch.int64),
        },
        os.path.join(args_cli.out, "engram_meta.safetensors"),
    )

    # capture hooks: per-layer output + next pre_mix, engram hashes, embedding
    cap: dict[str, torch.Tensor] = {}
    tag = {"phase": "prefill"}
    for i, layer in enumerate(model.layers):
        orig = layer.forward

        def fwd(x, start_pos, pre_mix, image_mask, *a, _i=i, _orig=orig):
            out, pm = _orig(x, start_pos, pre_mix, image_mask, *a)
            cap[f"{tag['phase']}.layer{_i:02d}.out"] = out[0].detach().cpu()
            cap[f"{tag['phase']}.layer{_i:02d}.pre_mix"] = pm[0].detach().cpu()
            return out, pm

        layer.forward = fwd

        # sublayer outputs and routing, to separate rounding noise from routing flips
        def attn_fwd(x, start_pos, *a, _i=i, _orig=layer.attn.forward):
            o = _orig(x, start_pos, *a)
            cap[f"{tag['phase']}.layer{_i:02d}.attn_out"] = o[0].detach().cpu()
            return o

        def moe_fwd(x, image_mask=None, _i=i, _orig=layer.ffn.forward):
            o = _orig(x, image_mask)
            cap[f"{tag['phase']}.layer{_i:02d}.moe_out"] = o[0].detach().cpu()
            return o

        def gate_fwd(x, image_mask=None, _i=i, _orig=layer.ffn.gate.forward):
            w, ix = _orig(x, image_mask)
            cap[f"{tag['phase']}.layer{_i:02d}.route_w"] = w.detach().float().cpu()
            cap[f"{tag['phase']}.layer{_i:02d}.route_ids"] = ix.detach().long().cpu()
            return w, ix

        layer.attn.forward, layer.ffn.forward, layer.ffn.gate.forward = attn_fwd, moe_fwd, gate_fwd
    orig_hash = eh.forward

    def hash_fwd(input_ids, start_pos, token_mask=None):
        h = orig_hash(input_ids, start_pos, token_mask)
        cap[f"{tag['phase']}.engram_hashes"] = h[0].detach().cpu()
        return h

    eh.forward = hash_fwd
    model.embed.register_forward_hook(lambda m, i, o: cap.__setitem__(f"{tag['phase']}.embed", o[0].detach().cpu()))

    prompt = encode_messages([{"role": "user", "content": args_cli.prompt}], thinking_mode="chat")
    ids = tokenizer.encode(prompt)
    print(f"prompt: {len(ids)} tokens", flush=True)
    tokens = torch.tensor([ids], dtype=torch.long)

    t0 = time.perf_counter()
    next_id, logits, _ = model.forward(tokens, 0)
    t_prefill = time.perf_counter() - t0
    cap["prefill.logits"] = logits[0].float().cpu()
    out_ids = [int(next_id[0])]
    print(f"prefill {t_prefill:.1f}s  first token {out_ids[0]!r} = {tokenizer.decode(out_ids)!r}", flush=True)

    step_times = []
    pos = len(ids)
    for step in range(args_cli.new_tokens - 1):
        if out_ids[-1] == tokenizer.eos_token_id:
            break
        tag["phase"] = "decode0" if step == 0 else "decode"
        t0 = time.perf_counter()
        next_id, logits, _ = model.forward(torch.tensor([[out_ids[-1]]], dtype=torch.long), pos)
        step_times.append(time.perf_counter() - t0)
        if step == 0:
            cap["decode0.logits"] = logits[0].float().cpu()
        pos += 1
        out_ids.append(int(next_id[0]))
        print(f"  token {len(out_ids):3d}: {tokenizer.decode(out_ids[-1:])!r}  ({step_times[-1]:.2f}s)", flush=True)

    golden = {k: v.contiguous() for k, v in cap.items() if not k.startswith("decode.")}
    golden["prompt_ids"] = torch.tensor(ids, dtype=torch.int64)
    golden["generated_ids"] = torch.tensor(out_ids, dtype=torch.int64)
    for n, (e, x, w, y) in enumerate(FIXTURE["records"]):
        golden[f"expert{n}.x"] = x.contiguous()
        golden[f"expert{n}.y"] = y.contiguous()
        golden[f"expert{n}.id"] = torch.tensor([args_cli.fixture_layer, e], dtype=torch.int64)
        if w is not None:
            golden[f"expert{n}.route_weight"] = w.float().contiguous()
    save_file(golden, os.path.join(args_cli.out, args_cli.golden_name), metadata={k: str(v) for k, v in overrides.items()})

    completion = tokenizer.decode(out_ids)
    run = {
        "golden": args_cli.golden_name,
        "overrides": overrides,
        "prompt": args_cli.prompt,
        "encoded_prompt": prompt,
        "completion": completion,
        "prompt_tokens": len(ids),
        "generated_tokens": len(out_ids),
        "seconds": {
            "build": round(t_build, 1),
            "load_trunk": round(t_load, 1),
            "prefill": round(t_prefill, 1),
            "decode_mean": round(sum(step_times) / max(len(step_times), 1), 2),
            "total": round(time.perf_counter() - t_start, 1),
        },
        "expert_cache": {"hits": CACHE.hits, "misses": CACHE.misses, "load_seconds": round(CACHE.load_seconds, 1)},
    }
    with open(os.path.join(args_cli.out, args_cli.golden_name.replace(".safetensors", ".run.json")), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, ensure_ascii=False)
    print("\ncompletion:", repr(completion))
    print(json.dumps(run["seconds"]), json.dumps(run["expert_cache"]))


if __name__ == "__main__":
    main()
