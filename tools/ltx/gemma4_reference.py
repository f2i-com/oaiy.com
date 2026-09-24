"""Official Gemma 4 Unified reference for offline validation of the Rust encoder."""
import argparse
import json
import math
import pathlib
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gemma', required=True)
parser.add_argument('--reference-deps', help='Isolated Transformers 5.17+ installation')
parser.add_argument('--mode', choices=['blocks', 'features'], default='blocks')
parser.add_argument('--output', default='target/ltx-golden')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()
if args.reference_deps:
    sys.path.insert(0, str(pathlib.Path(args.reference_deps).resolve()))
import torch
from safetensors import safe_open
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextDecoderLayer, Gemma4UnifiedTextModel, Gemma4UnifiedTextRotaryEmbedding,
)

path = pathlib.Path(args.gemma)
if path.suffix != '.safetensors':
    raise ValueError('Use a complete safetensors checkpoint')
weights = safe_open(str(path), framework='pt', device=args.device)
config = Gemma4UnifiedTextConfig(**json.loads(weights.metadata()['gemma_config'])['text_config'])
config._attn_implementation = 'sdpa'
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
with torch.device(args.device):
    rotary = Gemma4UnifiedTextRotaryEmbedding(config)

if args.mode == 'blocks':
    x = torch.sin(torch.arange(16 * 3840, device=args.device).float() * .013).reshape(1, 16, 3840).bfloat16()
    positions = torch.arange(1008, 1024, device=args.device).unsqueeze(0)
    mask = torch.full((16, 16), -torch.inf, device=args.device, dtype=torch.bfloat16).triu(1)[None, None]
    x.float().cpu().numpy().tofile(out / 'gemma4-input.f32')
    for index in [0, 11]:
        with torch.device('meta'):
            model = Gemma4UnifiedTextDecoderLayer(config, index)
        prefix = f'model.layers.{index}.'
        model.load_state_dict({key: weights.get_tensor(prefix + key) for key in model.state_dict()}, strict=True, assign=True)
        model.eval()
        with torch.inference_mode():
            pe = rotary(x, positions, config.layer_types[index])
            result = model(x.clone(), shared_kv_states={}, position_embeddings=pe, attention_mask=mask)
        result.float().cpu().numpy().tofile(out / f'gemma4-{index}-output.f32')
        print('Gemma 4 block', index, 'RMS', result.float().square().mean().sqrt().item(), flush=True)
        del model
else:
    with torch.device('meta'):
        model = Gemma4UnifiedTextModel(config)
    state = {key: weights.get_tensor('model.' + key) for key in model.state_dict()}
    model.load_state_dict(state, strict=True, assign=True)
    model.rotary_emb = rotary
    model.embed_tokens.embed_scale = torch.tensor(math.sqrt(3840), device=args.device)
    model.eval()
    ids = [2] + list(range(100, 115))
    tokens = torch.tensor([[0] * (1024 - len(ids)) + ids], device=args.device)
    with torch.inference_mode():
        result = model(input_ids=tokens, attention_mask=(tokens != 0).long(), output_hidden_states=True, use_cache=False)
        states = torch.stack([h[:, -len(ids):, :] for h in result.hidden_states], -1)
        assert states.shape[-1] == 49
        variance = states.square().mean(2, keepdim=True)
        features = (states * torch.rsqrt(variance + 1e-6)).reshape(1, len(ids), 3840 * 49) * math.sqrt(4096 / 3840)
    features.float().cpu().numpy().tofile(out / 'gemma4-features.f32')
    print('Gemma 4 complete features RMS', features.float().square().mean().sqrt().item(), flush=True)
