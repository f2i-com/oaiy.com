import torch,math,json,pathlib,copy
from safetensors import safe_open
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel,Gemma3RotaryEmbedding
import argparse
parser=argparse.ArgumentParser(description='Full official Gemma 3 hidden-feature reference for offline Rust verification.')
parser.add_argument('--gemma',required=True)
parser.add_argument('--config',required=True,help='Official Gemma 3 config.json')
parser.add_argument('--output',default='target/ltx-golden')
parser.add_argument('--device',default='cuda:0')
args=parser.parse_args()
cfg=json.load(open(args.config))['text_config'];cfg=Gemma3TextConfig(**cfg);cfg._attn_implementation='sdpa'
with torch.device('meta'):model=Gemma3TextModel(cfg)
f=safe_open(args.gemma,framework='pt',device=args.device)
state={k.removeprefix('model.'):f.get_tensor(k) for k in f.keys() if k.startswith('model.')};print(model.load_state_dict(state,strict=True,assign=True),flush=True)
model.rotary_emb=Gemma3RotaryEmbedding(cfg,device=args.device);local=copy.deepcopy(cfg);local.rope_theta=cfg.rope_local_base_freq;local.rope_scaling={'rope_type':'default'};model.rotary_emb_local=Gemma3RotaryEmbedding(local,device=args.device);model.embed_tokens.embed_scale=torch.tensor(math.sqrt(3840),device=args.device);model.eval()
ids=[2]+list(range(100,115));tokens=torch.tensor([[0]*(1024-len(ids))+ids],device=args.device);mask=(tokens!=0).long()
with torch.inference_mode():
 output=model(input_ids=tokens,attention_mask=mask,output_hidden_states=True,use_cache=False)
 states=torch.stack([h[:,-len(ids):,:] for h in output.hidden_states],-1)
 variance=states.square().mean(2,keepdim=True);features=(states*torch.rsqrt(variance+1e-6)).reshape(1,len(ids),3840*49)*math.sqrt(4096/3840)
out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True);features.float().cpu().numpy().tofile(out/'gemma-features.f32');print('Official Gemma full features',features.shape,'RMS',features.float().square().mean().sqrt().item(),flush=True)
