import sys,pathlib,json,torch
import argparse
parser=argparse.ArgumentParser(description='Official Lightricks VAE reference activations for offline Rust verification.')
parser.add_argument('--vae',required=True)
parser.add_argument('--source',required=True,help='Official LTX-2 repository checkout')
parser.add_argument('--reference-deps',help='Optional isolated Python dependency directory')
parser.add_argument('--output',default='target/ltx-golden')
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--encoder',action='store_true')
args=parser.parse_args()
sys.path.insert(0,str(pathlib.Path(args.source)/'packages/ltx-core/src'))
if args.reference_deps:sys.path.insert(0,args.reference_deps)
from ltx_core.model.video_vae.model_configurator import VideoDecoderConfigurator, VideoEncoderConfigurator
from safetensors import safe_open
p=args.vae
f=safe_open(p,framework='pt',device=args.device); metadata=f.metadata();metadata['config']=json.loads(metadata['config'])
component='encoder' if args.encoder else 'decoder'
with torch.device('meta'):model=(VideoEncoderConfigurator if args.encoder else VideoDecoderConfigurator).from_metadata(metadata)
state={}
for k in f.keys():
 key=k.removeprefix('vae.')
 if key.startswith((component+'.','per_channel_statistics.')):state[key.removeprefix(component+'.')]=f.get_tensor(k).to(torch.bfloat16)
print(model.load_state_dict(state,strict=True,assign=True),flush=True)
x=torch.sin(torch.arange(128*2*4*4,device=args.device,dtype=torch.float32)*.031).reshape(1,128,2,4,4).to(torch.bfloat16)
if args.encoder:x=torch.sin(torch.arange(3*128*128,device=args.device,dtype=torch.float32)*.031).reshape(1,3,1,128,128).to(torch.bfloat16)
with torch.inference_mode():y=model(x)
out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
x.float().cpu().numpy().tofile(out/('encoder-input.f32' if args.encoder else 'vae-input.f32'));y.float().cpu().numpy().tofile(out/('encoder-output.f32' if args.encoder else 'vae-output.f32'))
print('Official VAE',y.shape,'RMS',y.float().square().mean().sqrt().item(),flush=True)
