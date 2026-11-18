"""Read-only N053 CPU audit; run with PYTHONDONTWRITEBYTECODE=1."""
import ast, gc, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch, timm
from safetensors.torch import load_file
from timm.models._manipulate import adapt_input_conv
from timm.models.eva import checkpoint_filter_fn
ROOT='/home/hpc/v181bb/v181bb10/BSCCM'; W='/hnvme/workspace/v181bb10-bsccm'
sys.path.insert(0,ROOT)
from bsccm.encoders import LFBackbone,_snapshot
from bsccm.cytoland import CKPT,build_encoder
from bsccm.gpu_norm import GPUNorm
torch.set_num_threads(4); torch.manual_seed(0)
def emit(*args): print(*args,flush=True)
def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
emit('versions',torch.__version__,timm.__version__)
T=np.load(W+'/targets_v5cx_v10.npz'); has=(T['mask'].any(1)|(T['cls3']>=0))&(T['hist_row']<0); rows={}
for protocol,source in [('b1to0',1),('b0to1',0)]:
 tr=np.flatnonzero((T['batch']==source)&has); ev=np.flatnonzero((T['batch']!=source)&(T['split']==2)&has); rows[protocol]=tr,ev
 emit('rows',protocol,len(tr),len(ev),'overlap',len(np.intersect1d(tr,ev)),'sha',sha(tr),sha(ev))
 base=W+'/runs/n051_v10_base_'+('b10' if source else 'b01')
 emit('baseline args',json.load(open(base+'/result.json'))['args'])
 pred=np.load(base+'/eval_preds.npz'); emit('baseline G exact',np.array_equal(ev,pred['G']))
 ck=torch.load(base+'/ckpt.pt',map_location='cpu',weights_only=False)
 emit('baseline checkpoint rows',np.array_equal(tr,ck['tr']),np.array_equal(ev,ck['ev'])); del ck

g=np.array([rows['b1to0'][0][0],rows['b0to1'][0][0]])
raw=np.load(W+'/pack/led_array.npy',mmap_mode='r'); ph=np.load(W+'/pack/dpc.npy',mmap_mode='r'); gn=GPUNorm('cpu')
x=gn(torch.from_numpy(np.array(raw[g]).view(np.int16)),torch.from_numpy(np.array(ph[g])),torch.from_numpy(g))
emit('real input',g.tolist(),'range',x.min().item(),x.max().item(),'std',x.std(dim=(0,2,3)).tolist()); del gn,raw,ph
for arch in ('dinov3_s','dinov3_b','dinobloom_b','cytoland'):
 m=LFBackbone(arch).eval()
 if arch.startswith('dinov3'):
  repo={'dinov3_s':'vit_small_patch16_dinov3.lvd1689m','dinov3_b':'vit_base_patch16_dinov3.lvd1689m'}[arch]
  sd=checkpoint_filter_fn(load_file(_snapshot(repo)),m.net); sd['patch_embed.proj.weight']=adapt_input_conv(24,sd['patch_embed.proj.weight'])
  emit('DINO architecture',arch,'prefix',m.net.num_prefix_tokens,'registers',m.net.reg_token.shape[1],'rope',type(m.net.rope).__name__,'pool',m.net.global_pool)
 elif arch=='dinobloom_b':
  sd=torch.load(W+'/models/DinoBloom-B.pth',map_location='cpu',weights_only=True)['teacher']
  sd={k[9:]:v for k,v in sd.items() if k.startswith('backbone.') and k!='backbone.mask_token'}
  sd['patch_embed.proj.weight']=adapt_input_conv(24,sd['patch_embed.proj.weight']); emit('BLOOM pos',sd['pos_embed'].shape,'pool',m.net.global_pool)
 else:
  sd=torch.load(CKPT,map_location='cpu',weights_only=True)['state_dict']; sd['stem.0.weight']=sd['stem.0.weight'].repeat(1,24,1,1)/24
 actual=m.net.state_dict()
 emit('WEIGHTS',arch,'expected',len(sd),'actual',len(actual),'unequal',[k for k,v in sd.items() if k not in actual or not torch.equal(v,actual[k])],'extra',sorted(set(actual)-set(sd)))
 with torch.no_grad():
  y=m(x[:1]); yc=m(x[:1].contiguous(memory_format=torch.channels_last))
  with torch.autocast('cpu',dtype=torch.bfloat16): yb=m(x[:1].contiguous(memory_format=torch.channels_last))
 emit('FORWARD',arch,tuple(y.shape),'finite',bool(torch.isfinite(yb).all()),'layout maxdiff',(y-yc).abs().max().item(),'bf16 meanabs',(y-yb.float()).abs().mean().item())
 m.train()
 with torch.autocast('cpu',dtype=torch.bfloat16): loss=m(x[:1].contiguous(memory_format=torch.channels_last)).float().square().mean()
 loss.backward()
 emit('BACKWARD',arch,'nonfinite',[n for n,p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()],'gategrad',m.gate.g.grad.norm().item())
 emit('DECAY',arch,'gate norm tokens',[n for n,p in m.named_parameters() if n=='gate.g' or n.endswith('pos_embed') or n.endswith('reg_token') or n.endswith('norm.weight')][:8],'frozen',len([p for p in m.parameters() if not p.requires_grad]))
 del sd,actual,m,y,yc,yb,loss; gc.collect()

# Source-class import without heavyweight VisCy package/decode dependencies.
p=Path('/tmp/codex_n053rt/VisCy/packages/viscy-models/src/viscy_models/unet/fcmae.py'); tree=ast.parse(p.read_text())
tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('viscy_models'))]; ns={}; exec(compile(tree,str(p),'exec'),ns)
ref=ns['MaskedMultiscaleEncoder'](in_channels=1,stem_kernel_size=(1,2,2),in_stack_depth=1).eval()
orig=torch.load(W+'/models/cytoland/vscyto2d/epoch=399-step=23200.ckpt',map_location='cpu',weights_only=False)['state_dict']
orig={k[len('model.encoder.'):]:v for k,v in orig.items() if k.startswith('model.encoder.')}; emit('CYTO original strict',ref.load_state_dict(orig))
conv=build_encoder(1,0).eval(); expected={}
for k,v in orig.items():
 if k.startswith('stem.conv3d'): continue
 k=k.replace('stem.conv2d','stem.0').replace('stem.norm','stem.1').replace('.dwconv.','.conv_dw.').replace('.layernorm.','.norm.'); expected[k]=v
emit('CYTO original mapping unequal',[k for k,v in expected.items() if not torch.equal(v,conv.state_dict()[k])],'dw biases',[(n,p.abs().max().item(),p.requires_grad) for n,p in conv.named_parameters() if n.endswith('conv_dw.bias')])
emit('CYTO eps ref',[(n,mod.eps) for n,mod in ref.named_modules() if hasattr(mod,'eps')])
with torch.no_grad():
 xr=x[:1,-1:]; r,_=ref(xr.unsqueeze(2)); t=conv.forward_features(xr)
 emit('CYTO feature diff before eps fix',(r[-1]-t).abs().max().item(),(r[-1]-t).abs().mean().item())
 for stage in conv.stages:
  if isinstance(stage.downsample,torch.nn.Sequential): stage.downsample[0].eps=1e-6
 t=conv.forward_features(xr); emit('CYTO feature diff after eps fix',(r[-1]-t).abs().max().item(),(r[-1]-t).abs().mean().item())
 repeat=build_encoder(24,0).eval(); emit('CYTO repeat stem',(conv.stem(xr)-repeat.stem(xr.repeat(1,24,1,1))).abs().max().item())
q=torch.nn.Parameter(torch.ones(1)); z=torch.nn.Parameter(torch.ones(1)); frozen=torch.nn.Parameter(torch.ones(1),requires_grad=False)
o=torch.optim.AdamW([{'params':[q,frozen],'lr':5e-5},{'params':[z],'lr':5e-4}],weight_decay=.05)
s=torch.optim.lr_scheduler.OneCycleLR(o,[5e-5,5e-4],total_steps=100,pct_start=.1); ratios=[]
for i in range(40):
 ratios.append(o.param_groups[0]['lr']/o.param_groups[1]['lr']); q.grad=torch.zeros_like(q); z.grad=torch.zeros_like(z); o.step(); s.step()
o2=torch.optim.AdamW([{'params':[q,frozen]},{'params':[z]}],weight_decay=.05); s2=torch.optim.lr_scheduler.OneCycleLR(o2,[5e-5,5e-4],total_steps=100,pct_start=.1)
o2.load_state_dict(o.state_dict()); s2.load_state_dict(s.state_dict())
emit('SCHED ratio',min(ratios),max(ratios),'resume',o2.param_groups[0]['lr']==o.param_groups[0]['lr'],s2.state_dict()==s.state_dict(),'frozen unchanged',frozen.item(),id(frozen) in {id(p) for p in o.state})
torch.manual_seed(0); dummy=torch.nn.Linear(20,20); first=torch.randperm(20); second=torch.randperm(20)
torch.manual_seed(0); dummy=torch.nn.Linear(20,20); resumed=torch.randperm(20)
emit('RESUME RNG repeats first',torch.equal(first,resumed),'matches uninterrupted second',torch.equal(second,resumed))
