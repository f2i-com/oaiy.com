import sys,pathlib,json,torch
import argparse
parser=argparse.ArgumentParser(description='Official Lightricks VAE reference activations for offline Rust verification.')
parser.add_argument('--vae',required=True)
parser.add_argument('--source',required=True,help='Official LTX-2 repository checkout')
parser.add_argument('--reference-deps',help='Optional isolated Python dependency directory')
parser.add_argument('--output',default='target/ltx-golden')
parser.add_argument('--device',default='cuda:0')
args=parser.parse_args()
sys.path.insert(0,str(pathlib.Path(args.source)/'packages/ltx-core/src'))
if args.reference_deps:sys.path.insert(0,args.reference_deps)
from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator
from safetensors import safe_open
p=args.vae
f=safe_open(p,framework='pt',device=args.device); metadata=f.metadata();metadata['config']=json.loads(metadata['config'])
with torch.device('meta'):model=VideoDecoderConfigurator.from_metadata(metadata)
state={k.removeprefix('decoder.'):f.get_tensor(k).to(torch.bfloat16) for k in f.keys() if k.startswith(('decoder.','per_channel_statistics.'))}
print(model.load_state_dict(state,strict=True,assign=True),flush=True)
x=torch.sin(torch.arange(128*2*4*4,device=args.device,dtype=torch.float32)*.031).reshape(1,128,2,4,4).to(torch.bfloat16)
with torch.inference_mode():y=model(x)
out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
x.float().cpu().numpy().tofile(out/'vae-input.f32');y.float().cpu().numpy().tofile(out/'vae-output.f32')
print('Official VAE',y.shape,'RMS',y.float().square().mean().sqrt().item(),flush=True)
