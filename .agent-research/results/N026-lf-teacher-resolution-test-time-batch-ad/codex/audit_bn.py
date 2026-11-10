"""CPU-only diagnostic; does NOT overwrite models or evaluate a replacement headline."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast,contextlib,copy,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from types import SimpleNamespace
sys.path.insert(0,'/home/hpc/v181bb/v181bb10/BSCCM')
from bsccm.data import LFNorm
from bsccm.gpu_norm import GPUNorm,RawSet
W=Path('/hnvme/workspace/v181bb10-bsccm'); torch.set_num_threads(1); torch.set_num_interop_threads(1)
# Execute actual production fwd AST on a tiny encoder; CUDA autocast replaced with no-op.
source=Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py').read_text(); tr=ast.parse(source)
node=next(n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name=='fwd')
class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.bn=nn.BatchNorm2d(48)
    def forward(self,x): return self.bn(x).mean((-1,-2))
model=nn.ModuleDict({'enc':Toy(),'reg':nn.Linear(96,13),'c3':nn.Linear(96,3),'c10':nn.Linear(96,10)})
ns=dict(torch=SimpleNamespace(bfloat16=torch.bfloat16,cat=torch.cat,ones_like=torch.ones_like,autocast=lambda *a,**kw:contextlib.nullcontext(),channels_last=torch.channels_last),F=F,SSL=True,a=SimpleNamespace(dual_zoom=2),model=model)
exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual_fwd>','exec'),ns)
x=torch.zeros(4,24,128,128); x[:,:,32:96,32:96]=4
out=ns['fwd'](x)
print('PRODUCTION_FWD toy full mean=1 zoom mean=4 BN running mean=',model['enc'].bn.running_mean[:2].tolist(),'BN count=',model['enc'].bn.num_batches_tracked.item(),'head_shapes',[tuple(z.shape) for z in out],flush=True)
print('SSL ones channel running mean',model['enc'].bn.running_mean[24:].unique().tolist(),'(expected .19 after two updates)',flush=True)
T=dict(np.load(W/'targets_v5.npz',allow_pickle=True)); norm=LFNorm(); common=[j for j,n in enumerate(T['names'][:12]) if n!='grp:CD123/HLA-DR/CD14']
all_results=[]
def stats(P,G):
    rs=[]; counts=[]
    for j in common:
        m=T['mask'][G,j]; counts.append(int(m.sum())); rs.append(float(np.corrcoef(P[m,j],T['Y'][G[m],j])[0,1]))
    return dict(r=float(np.mean(rs)),counts=counts,per_target=rs)
def sample(rows,n,seed):
    # Equal counts within each slide guarantee usable per-target masks in small diagnostic.
    rng=np.random.default_rng(seed); return np.concatenate([rng.choice(rows[T['slide'][rows]==s],min(n,(T['slide'][rows]==s).sum()),replace=False) for s in np.unique(T['slide'][rows])])
def inputs(rows):
    for start in range(0,len(rows),16):
        g=rows[start:start+16]; x=torch.from_numpy(np.stack([norm(int(i),list(range(23)),True) for i in g])); x=torch.cat([x,torch.ones_like(x)],1)
        xz=F.interpolate(x[...,32:96,32:96],size=(128,128),mode='bilinear',align_corners=False)
        yield x,xz
@torch.no_grad()
def evaluate(enc,head,rows,banks=None):
    enc.eval(); P=[]
    for x,xz in inputs(rows):
        if banks: set_bn(enc,banks[0])
        h=enc(x)
        if banks: set_bn(enc,banks[1])
        hz=enc(xz); P.append(head(torch.cat([h,hz],1)).numpy())
    return np.concatenate(P)
def bn_state(enc): return {k:v.clone() for k,v in enc.state_dict().items() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
def set_bn(enc,bank):
    state=enc.state_dict()
    for k,v in bank.items(): state[k].copy_(v)
for d in ('b10','b01'):
    t0=time.monotonic(); S=torch.load(W/'runs'/f'n026_dual2_{d}'/'ckpt.pt',map_location='cpu',weights_only=False)
    enc=resnet50(weights=None); enc.conv1=nn.Conv2d(48,64,7,2,3,bias=False); enc.fc=nn.Identity()
    weights={k.removeprefix('enc.'):v for k,v in S['m'].items() if k.startswith('enc.')}
    print('STRICT torchvision checkpoint load',d,enc.load_state_dict(weights,strict=True),flush=True)
    head=nn.Linear(4096,13); head.load_state_dict({k.removeprefix('reg.'):v for k,v in S['m'].items() if k.startswith('reg.')})
    counts=[int(v) for k,v in weights.items() if k.endswith('num_batches_tracked')]
    ssl=torch.load(W/'runs/n007_ssl_aug/encoder_final.pt',map_location='cpu',weights_only=True)
    inherited=[int(v) for k,v in ssl.items() if k.endswith('num_batches_tracked')]; steps=(len(S['tr'])//256)*10
    print('BN update counts',d,'actual unique',sorted(set(counts)),'SSL unique',sorted(set(inherited)),'expected SSL+2*steps',inherited[0]+2*steps,flush=True)
    # Check LFNorm CPU reference against actual GPUNorm CPU on just two cells.
    if d=='b10':
        rs=RawSet(S['ev'][:2]); items=[rs[i] for i in range(2)]; raw=torch.stack([z[0] for z in items]); ph=torch.stack([z[1] for z in items]); g=torch.tensor([z[2] for z in items])
        gnorm=GPUNorm('cpu'); a=gnorm(raw,ph,g,'all',True); ref=torch.from_numpy(np.stack([norm(int(i),list(range(23)),True) for i in g])); print('NORM CPU reference vs production maxabs',float((a-ref).abs().max()),flush=True); del gnorm
    ev=sample(S['ev'],48,2600); tr=sample(S['tr'][T['split'][S['tr']]==0],32,2601)
    original=evaluate(enc,head,ev); saved=np.load(W/'runs'/f'n026_dual2_{d}'/'eval_preds.npz')['P']; ix=np.searchsorted(S['ev'],ev)
    print('ORIGINAL',d,'n_eval',len(ev),'n_source_calibration',len(tr),'score',stats(original,ev),'saved_gpu_score',stats(saved[ix],ev),'pred_corr',float(np.corrcoef(original.ravel(),saved[ix].ravel())[0,1]),flush=True)
    banks=[]
    for view in (0,1):
        enc.train()
        for m in enc.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm): m.reset_running_stats(); m.momentum=None
        with torch.no_grad():
            for pair in inputs(tr): enc(pair[view])
        banks.append(bn_state(enc))
    corrected=evaluate(enc,head,ev,banks)
    result=dict(direction=d,n_eval=len(ev),n_calibration=len(tr),original=stats(original,ev),saved=stats(saved[ix],ev),separate_source_bn=stats(corrected,ev),seconds=time.monotonic()-t0)
    all_results.append(result); print('SEPARATE_SOURCE_BN',json.dumps(result),flush=True)
    del S,enc,head
json.dump(all_results,open(Path(__file__).with_name('audit_bn_metrics.json'),'w'),indent=2)
