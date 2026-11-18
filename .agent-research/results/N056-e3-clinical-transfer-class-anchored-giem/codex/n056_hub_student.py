#!/usr/bin/env -S python3 -I -B
"""Paired N056 A3 hub and anchored Giemsa student. See report.md."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path('/hnvme/workspace/v181bb10-bsccm')
REPO = Path('/home/hpc/v181bb/v181bb10/BSCCM')
ANCH = ['lymphocyte', 'neutrophil', 'monocyte']
STOP = False

def sibling(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

core = sibling('n056_hub_core')
legacy = sibling('n056_legacy')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stop(signum, frame):
    global STOP
    STOP = True
    print('SIGTERM: checkpoint at next optimizer boundary', flush=True)

def atomic(obj, path):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp); os.replace(tmp, path)

def write_json(path, obj):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore(s):
    random.setstate(s['py']); np.random.set_state(s['np']); torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])

def pairs(root=ROOT):
    out = []
    for i, name in enumerate(['n029_tt6c_s0', 'n032_tt6c_t1_s0', 'n032_tt6c_t2_s0']):
        path = Path(root) / 'runs' / name
        a = json.loads((path / 'result.json').read_text())['args']
        expected = dict(holdout='fov', anchor_ce=.5, target_half='A', seed=0, steps=3000,
                        targets='targets_v4c.npz', aux_targets='targets_v6c.npz', freeze=8,
                        dim=256, bs=128, lr=.0001, dann=0., aug=1, stain_norm='none')
        for k, v in expected.items():
            if a[k] != v: raise ValueError(f'{name}: {k} differs from preregistration')
        for key in ['lf_feats', 'aux_feats']:
            if not (Path(a[key]) / 'lf_feats_all.npy').exists(): raise FileNotFoundError(a[key])
        out.append(dict(teacher=f't{i}', checkpoint=str(path/'gs_bind.pt'), args=a,
                        config_sha=sha(path/'result.json')))
    return out

class Data:
    def __init__(self, a, pair):
        self.a, self.root = a, Path(a.data)
        with np.load(self.root/'targets_v4c.npz') as t:
            self.t = {k: t[k].copy() for k in ['Y','mask','T','names','hist_row','split','batch','slide','cls3','cls3_names']}
        with np.load(self.root/'targets_v6c.npz') as u:
            for key in ['hist_row','split','batch','slide','names']:
                assert np.array_equal(self.t[key], u[key]), key
            self.y = self.t['Y'].copy(); self.m = self.t['mask'].copy()
            self.y[:,-1], self.m[:,-1] = u['Y'][:,-1], u['mask'][:,-1]
            self.autofl_y, self.autofl_m = u['Y'][:,-1].copy(), u['mask'][:,-1].copy()
        self.names = [str(n) for n in self.t['names']]
        h, sp = self.t['hist_row'], self.t['split']
        self.lf_train = np.flatnonzero((h<0)&(sp==0)&self.m.any(1))
        self.gs_train = np.flatnonzero((h>=0)&(sp==0))
        self.gs_test = np.flatnonzero((h>=0)&(sp==2))
        self.gidx = np.flatnonzero(h>=0); self.gidx = self.gidx[np.argsort(h[self.gidx])]
        assert np.array_equal(h[self.gidx],np.arange(len(self.gidx)))
        self.features = [np.load(Path(pair['args'][key])/'lf_feats_all.npy',mmap_mode='r') for key in ['lf_feats','aux_feats']]
        # Match the legacy LF normalization exactly (same rows for each teacher).
        normrows = np.flatnonzero((h<0)&(sp==0)&self.t['mask'][:,:12].any(1))
        if a.tiny: normrows = normrows[:96]
        self.mu, self.sd = [], []
        for x in self.features:
            assert x.shape == (len(h),512)
            xx = torch.from_numpy(np.asarray(x[normrows],np.float32).copy())
            self.mu.append(xx.mean(0).numpy()); self.sd.append((xx.std(0)+1e-6).numpy())
        self.mu, self.sd = np.concatenate(self.mu), np.concatenate(self.sd)
        self.hist = np.load(self.root/'pack/histology.npy',mmap_mode='r')
        self.scale = float(np.percentile(self.hist[:500:5],99.9))  # legacy paired preprocessing
        p = REPO/'.agent-research/results/N008-external-lineage-anchor-pbc-acevedo-mgg/gs_lf_cls3_pred.npz'
        self.gs_cls = np.load(p)['pred'].astype(np.int64)
        assert len(self.gs_cls)==len(self.gidx)
        self.anchors = legacy.ExtSet(pair['args']['target_zip'],'A',ANCH)
        if a.tiny:
            # Engineering-only CPU path: strict source split filtering is retained.
            self.lf_train = np.random.default_rng(56).permutation(self.lf_train)[:96]
            self.gs_train = self.gs_train[:8]; self.gs_test=self.gs_test[:4]
        assert not np.intersect1d(self.gs_train,self.gs_test).size
    def x(self,rows,dev):
        x=np.concatenate([np.asarray(f[rows],np.float32) for f in self.features],axis=1)
        return torch.as_tensor((x-self.mu)/self.sd,device=dev)
    def images(self,rows,dev,train):
        x=torch.from_numpy(np.asarray(self.hist[self.t['hist_row'][rows]],np.int16).copy())
        return legacy.pipeline((x.to(dev).float()/self.scale).clamp(0,1).permute(0,3,1,2),train)

def setup_pipeline(dev):
    legacy.dev=dev; legacy.a=SimpleNamespace(stain_norm='none',aug=1)
    legacy.MEAN=torch.tensor([.485,.456,.406],device=dev).view(1,3,1,1)
    legacy.STD=torch.tensor([.229,.224,.225],device=dev).view(1,3,1,1)

def legacy_mlp(i,o):
    return nn.Sequential(nn.Linear(i,1024),nn.GELU(),nn.LayerNorm(1024),nn.Linear(1024,o))

def student(tiny=False,pretrained=False,data=ROOT):
    import timm
    if tiny:
        vit=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=0,depth=1)
    else:
        vit=timm.create_model('vit_base_patch14_dinov2',pretrained=False,num_classes=0,img_size=224)
        if pretrained:
            state=torch.load(Path(data)/'models/DinoBloom-B.pth',map_location='cpu',weights_only=True)['teacher']
            vit.load_state_dict({k[9:]:v for k,v in state.items() if k.startswith('backbone.') and k!='backbone.mask_token'},strict=True)
    vit.patch_embed.requires_grad_(False); vit.cls_token.requires_grad=False; vit.pos_embed.requires_grad=False
    for i,block in enumerate(vit.blocks): block.requires_grad_(tiny or i>=8)
    return nn.ModuleDict(dict(vit=vit,PG=legacy_mlp(vit.num_features,256),C=legacy_mlp(256,3)))

def encode(net,x):
    dev=x.device.type
    with torch.autocast('cuda',torch.bfloat16) if dev=='cuda' else contextlib.nullcontext(): h=net['vit'](x).float()
    return F.normalize(net['PG'](h),dim=-1)

def train(a):
    global STOP
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pair=pairs(a.data)[a.teacher]; write_json(out/'pair.json',pair)
    contract=dict(args=vars(a),pair=pair,report=sha(HERE/'report.md'),
                  code={p.name:sha(p) for p in HERE.glob('n056_*.py')})
    done=out/'done.json'
    if done.exists():
        completed=json.loads(done.read_text())
        if completed.get('contract')!=contract: raise ValueError('completed run contract changed')
        if not (out/'hub_student.pt').exists(): raise FileNotFoundError('completed artifact missing')
        print('completed run: immediate exit',flush=True); return
    dev=torch.device(a.device); setup_pipeline(dev); seed_all(0)
    d=Data(a,pair)
    # Full model/optimizer/RNG cursor atomically saved at every 100 updates.
    ckpath=out/'checkpoint.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False) if ckpath.exists() else None
    if ck and ck['contract']!=contract: raise ValueError('resume contract changed')
    hub=core.Spaces(256).to(dev)
    net=student(a.tiny,True,a.data).to(dev)
    if ck:
        hub.load_state_dict(ck['hub']); net.load_state_dict(ck['student'])
    for stage,total in [('hub',a.hub_steps),('student',a.steps)]:
        if ck and ck['stage']=='student' and stage=='hub': continue
        module=hub if stage=='hub' else net
        if stage=='student': hub.eval().requires_grad_(False)
        opt=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=1e-4,weight_decay=.05)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,1e-4,total_steps=total,pct_start=.05 if total>=100 else .5)
        start=0
        if ck and ck['stage']==stage:
            opt.load_state_dict(ck['opt'])
            for state in opt.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v): state[k]=v.to(dev)
            sched.load_state_dict(ck['sched']); start=ck['step']; restore(ck['rng'])
        def save(step):
            atomic(dict(contract=contract,stage=stage,step=step,hub=hub.state_dict(),student=net.state_dict(),
                        opt=opt.state_dict(),sched=sched.state_dict(),rng=rng()),ckpath)
        print(f'ready {stage} at {start}/{total}',flush=True)
        for step in range(start,total):
            if STOP: save(step); raise SystemExit(143)
            # Independent stateless batch selection permits exact resume without loader cursor ambiguity.
            rr=np.random.default_rng(56000+step).choice(d.lf_train if stage=='hub' else d.gs_train,
                      min(a.hub_bs if stage=='hub' else a.bs,len(d.lf_train if stage=='hub' else d.gs_train)),replace=False)
            module.train()
            if stage=='hub':
                y=torch.as_tensor(d.y[rr],device=dev); m=torch.as_tensor(d.m[rr],device=dev)
                y=torch.where(m,y,0); z=hub.lf(d.x(rr,dev)); zm=hub.marker(y,m)
                loss=core.mse_mask(hub.decode(z),y,m)+hub.sig(z,zm,core.near_duplicate(y,m,.25))
            else:
                zg=encode(net,d.images(rr,dev,True))
                with torch.no_grad(): zl=hub.lf(d.x(rr,dev))
                dist=(zg-zl).square().sum(-1).mean()
                ids=np.random.default_rng(156000+step).choice(len(d.anchors),len(rr),replace=False)
                items=[d.anchors[int(i)] for i in ids]
                xa=torch.stack([v[0] for v in items]); ya=torch.tensor([v[1] for v in items],device=dev)
                za=encode(net,legacy.pipeline((xa.to(dev).float()/255.).permute(0,3,1,2),True))
                source_class=torch.as_tensor(d.gs_cls[d.t['hist_row'][rr]],device=dev)
                ce=F.cross_entropy(net['C'](za*16),ya)+F.cross_entropy(net['C'](zg*16),source_class)
                loss=dist+.5*ce
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(module.parameters(),1)
            opt.step(); sched.step()
            if (step+1)%100==0 or step+1==total:
                print(json.dumps(dict(stage=stage,step=step+1,loss=float(loss.detach()))),flush=True); save(step+1)
            if STOP: save(step+1); raise SystemExit(143)
        if stage=='hub':
            atomic(dict(state=hub.state_dict(),mu=d.mu,sd=d.sd,contract=contract),out/'hub.pt')
            ck=None
    artifact=dict(hub=hub.state_dict(),student=net.state_dict(),scale=d.scale,names=d.names,
                  gs_train=d.gs_train,lf_train=d.lf_train,mu=d.mu,sd=d.sd,contract=contract,tiny=a.tiny)
    atomic(artifact,out/'hub_student.pt')
    write_json(out/'done.json',dict(done=True,checkpoint=str(out/'hub_student.pt'),tiny=a.tiny,contract=contract))
    if STOP: raise SystemExit(143)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',action='store_true'); p.add_argument('--out',required=True)
    p.add_argument('--data',default=str(ROOT)); p.add_argument('--teacher',type=int,choices=[0,1,2],default=0)
    p.add_argument('--device',default='cuda'); p.add_argument('--tiny',action='store_true')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--hub-steps',type=int,default=3000)
    p.add_argument('--bs',type=int,default=128); p.add_argument('--hub-bs',type=int,default=512)
    p.add_argument('--threads',type=int,default=16)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGUSR1,stop)
    if a.manifest:
        Path(a.out).mkdir(parents=True,exist_ok=True); write_json(Path(a.out)/'pairs.json',pairs(a.data)); return
    if a.tiny and a.device!='cpu': raise ValueError('tiny is a CPU engineering check')
    if not a.tiny and (a.steps,a.hub_steps,a.bs,a.hub_bs)!=(3000,3000,128,512):
        raise ValueError('production configuration differs from report.md')
    train(a)
if __name__=='__main__': main()
