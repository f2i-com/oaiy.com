import torch,math,pathlib
from safetensors import safe_open
import argparse
parser=argparse.ArgumentParser(description='Offline Gemma 3 reference activations; not an inference runtime.')
parser.add_argument('--gemma',required=True)
parser.add_argument('--output',default='target/ltx-golden')
parser.add_argument('--device',default='cuda:0')
args=parser.parse_args()
out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
f=safe_open(args.gemma,framework='pt',device=args.device)
n=16
x=torch.sin(torch.arange(n*3840,dtype=torch.float32,device=args.device)*0.013).reshape(1,n,3840).to(torch.bfloat16)
x.float().cpu().numpy().tofile(out/'gemma-input.f32')
def norm(x,w):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)*(1+w.float())).to(x.dtype)
for i in [0,5]:
 prefix=f'model.layers.{i}.'
 def weight(k):return f.get_tensor(prefix+k)
 def linear(x,k):return torch.nn.functional.linear(x,weight(k+'.weight'))
 def rms(x,k):return norm(x,weight(k+'.weight'))
 h=rms(x,'input_layernorm')
 q=linear(h,'self_attn.q_proj').reshape(1,n,16,256).transpose(1,2)
 k=linear(h,'self_attn.k_proj').reshape(1,n,8,256).transpose(1,2)
 v=linear(h,'self_attn.v_proj').reshape(1,n,8,256).transpose(1,2)
 q=rms(q,'self_attn.q_norm');k=rms(k,'self_attn.k_norm')
 theta=1e6 if i==5 else 1e4;factor=8 if i==5 else 1
 inv=1/(theta**(torch.arange(0,256,2,device=x.device).float()/256))/factor
 angles=torch.arange(n,device=x.device).float()[:,None]*inv[None,:]
 c=angles.cos().to(x.dtype)[None,None];s=angles.sin().to(x.dtype)[None,None]
 def rope(t):a,b=t.chunk(2,-1);return torch.cat([a*c-b*s,b*c+a*s],-1)
 q=rope(q);k=rope(k).repeat_interleave(2,1);v=v.repeat_interleave(2,1)
 a=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(1,n,4096)
 y=x+rms(linear(a,'self_attn.o_proj'),'post_attention_layernorm')
 h=rms(y,'pre_feedforward_layernorm')
 h=torch.nn.functional.gelu(linear(h,'mlp.gate_proj'),approximate='tanh')*linear(h,'mlp.up_proj')
 y=y+rms(linear(h,'mlp.down_proj'),'post_feedforward_layernorm')
 y.float().cpu().numpy().tofile(out/f'gemma-layer-{i}.f32')
 print(i,'rms',y.float().square().mean().sqrt().item(),flush=True)
