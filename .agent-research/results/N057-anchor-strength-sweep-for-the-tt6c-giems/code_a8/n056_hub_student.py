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

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
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

def pairs(root=ROOT, seed=0):
    out = []
    for i in range(3):
        if seed not in range(3 if i==0 else 2):
            out.append(None); continue
        name = f'n029_tt6c_s{seed}' if i==0 else f'n032_tt6c_t{i}_s{seed}'
        _ov = Path(__file__).with_name('t_override.json')   # N057: alternative T checkpoints (anchor-strength sweep)
        _ov = json.loads(_ov.read_text()) if _ov.exists() else None
        if _ov: name = _ov['name'].format(i=i, seed=seed)
        path = Path(root) / 'runs' / name
        a = json.loads((path / 'result.json').read_text())['args']
        expected = dict(holdout='fov', anchor_ce=(_ov['anchor_ce'] if _ov else .5), target_half='A', seed=seed, steps=3000,
                        targets='targets_v4c.npz', aux_targets='targets_v6c.npz', freeze=8,
                        dim=256, bs=128, lr=.0001, dann=0., aug=1, stain_norm='none',w_distill=1.)
        for k, v in expected.items():
            if a[k] != v: raise ValueError(f'{name}: {k} differs from preregistration')
        if a.get('w_synth',0.)!=0. or a.get('synth',''):raise ValueError('unexpected synthetic supervision')
        for key in ['lf_feats', 'aux_feats']:
            if not (Path(a[key]) / 'lf_feats_all.npy').exists(): raise FileNotFoundError(a[key])
        out.append(dict(teacher=f't{i}', checkpoint=str(path/'gs_bind.pt'), seed=seed, args=a,
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
            self.m[(self.t['batch']==1),8] = False  # N054/K019 group-8 draw-2 QC
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
        drop = sibling('n056_duplicates').excluded(pair['args']['target_zip'])
        keep = [i for i,n in enumerate(self.anchors.names) if n not in drop]
        self.anchors.names = [self.anchors.names[i] for i in keep]
        self.anchors.cls = [self.anchors.cls[i] for i in keep]
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

def student(tiny=False,pretrained=False,data=ROOT,reference=False):
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
    if reference:
        return nn.ModuleDict(dict(vit=vit,PG=legacy_mlp(vit.num_features,256),PL=legacy_mlp(512,256),
                                 S=legacy_mlp(256,13),D=legacy_mlp(vit.num_features,1),C=legacy_mlp(256,3)))
    return nn.ModuleDict(dict(vit=vit,PG=legacy_mlp(vit.num_features,256),C=legacy_mlp(256,3)))

def encode(net,x):
    dev=x.device.type
    with torch.autocast('cuda',torch.bfloat16) if dev=='cuda' else contextlib.nullcontext(): h=net['vit'](x).float()
    return F.normalize(net['PG'](h),dim=-1)

class StudentSet(torch.utils.data.Dataset):
    """Index is a stateless (global row, anchor index) pair from StepSampler."""
    def __init__(self,d):self.d=d
    def __len__(self):return len(self.d.gs_train)
    def __getitem__(self,key):
        row,anchor=key
        x=torch.from_numpy(np.array(self.d.hist[self.d.t['hist_row'][row]],dtype=np.int16))
        xa,ya=self.d.anchors[anchor]
        return x,row,xa,ya

class StepSampler:
    def __init__(self,d,a,start):self.d,self.a,self.start=d,a,start
    def __iter__(self):
        for step in range(self.start,self.a.steps):
            rg=np.random.default_rng(56000+1000000*self.a.seed+step)
            rows=rg.choice(self.d.gs_train,min(self.a.bs,len(self.d.gs_train)),replace=False)
            anchors=rg.choice(len(self.d.anchors),len(rows),replace=False)
            yield [(int(r),int(i)) for r,i in zip(rows,anchors)]
    def __len__(self):return self.a.steps-self.start

def student_loader(d,a,start):
    # Image decoding happens in 12 workers; loader seeds never consume model RNG.
    return torch.utils.data.DataLoader(StudentSet(d),batch_sampler=StepSampler(d,a,start),
        num_workers=a.workers,generator=torch.Generator().manual_seed(560),
        persistent_workers=a.workers>0,pin_memory=a.device=='cuda')

def shared_blocks(net):
    return [p for block in net['vit'].blocks for p in block.parameters() if p.requires_grad]

def gradient_norms(terms,params):
    out={}
    for name,loss in terms.items():
        gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        out[name]=math.sqrt(sum(float(g.detach().float().square().sum()) for g in gs if g is not None))
    return out

def matched_weight(share,nonanchor,anchor):
    if not 0<share<1 or min(nonanchor,anchor)<=1e-12:raise ValueError('degenerate gradient budget')
    return share/(1-share)*nonanchor/anchor

class TeacherReplay:
    """Original train_gs_bind objective; same augmented tensors as H, no clinical input."""
    def __init__(self,a,d,dev):
        self.d,self.dev=d,dev
        devices=[dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(a.seed)
            self.net=student(a.tiny,True,a.data,reference=True).to(dev)
        self.tsig=nn.Parameter(torch.tensor(math.log(10.),device=dev));self.bsig=nn.Parameter(torch.tensor(-10.,device=dev))
        self.params=[p for p in self.net.parameters() if p.requires_grad]+[self.tsig,self.bsig]
        self.opt=torch.optim.AdamW(self.params,lr=1e-4,weight_decay=.05)
        self.sched=torch.optim.lr_scheduler.OneCycleLR(self.opt,1e-4,total_steps=a.steps,pct_start=.05 if a.steps>=100 else .5)
        # Fit original per-target ridge exactly, including original target masks.
        self.teach=torch.empty((len(d.gidx),13),device=dev)
        with torch.no_grad():
            for j in range(13):
                yy=d.autofl_y if j==12 else d.t['Y'][:,j]
                mm=d.autofl_m if j==12 else d.t['mask'][:,j]
                rows=np.flatnonzero((d.t['hist_row']<0)&(d.t['split']==0)&mm)
                rows=np.random.default_rng(j).permutation(rows)[:(96 if a.tiny else 60000)]
                cols=slice(512,1024) if j==12 else slice(0,512)
                x=d.x(rows,dev)[:,cols];xb=torch.cat([x,torch.ones(len(x),1,device=dev)],1)
                reg=10*torch.eye(xb.shape[1],device=dev);reg[-1,-1]=0
                w=torch.linalg.solve(xb.T@xb+reg,xb.T@torch.as_tensor(yy[rows],device=dev))
                gx=d.x(d.gidx,dev)[:,cols]
                self.teach[:,j]=gx@w[:-1]+w[-1]
    def update(self,xg,xa,rows,ya):
        self.net.train();zg=encode(self.net,xg);za=encode(self.net,xa)
        zl=F.normalize(self.net['PL'](self.d.x(rows,self.dev)[:,:512]),dim=-1)
        logits=zg@zl.T*self.tsig.exp()+self.bsig
        labels=2*torch.eye(len(rows),device=self.dev)-1
        sig=-F.logsigmoid(labels*logits).sum()/len(rows)
        dist=F.mse_loss(self.net['S'](zg*16),self.teach[self.d.t['hist_row'][rows]])
        cls=torch.as_tensor(self.d.gs_cls[self.d.t['hist_row'][rows]],device=self.dev)
        ce=F.cross_entropy(self.net['C'](za*16),ya)+F.cross_entropy(self.net['C'](zg*16),cls)
        norms=gradient_norms(dict(sig=sig,dist=dist,anchor=ce),shared_blocks(self.net))
        self.opt.zero_grad(set_to_none=True);(sig+dist+.5*ce).backward()
        nn.utils.clip_grad_norm_(self.params,1);self.opt.step();self.sched.step()
        return norms
    def state(self):
        return dict(net=self.net.state_dict(),tsig=self.tsig.detach(),bsig=self.bsig.detach(),
                    opt=self.opt.state_dict(),sched=self.sched.state_dict())
    def load(self,state):
        self.net.load_state_dict(state['net'])
        with torch.no_grad():self.tsig.copy_(state['tsig']);self.bsig.copy_(state['bsig'])
        self.opt.load_state_dict(state['opt']);self.sched.load_state_dict(state['sched'])
        for item in self.opt.state.values():
            for k,v in item.items():
                if torch.is_tensor(v):item[k]=v.to(self.dev)


def train(a):
    global STOP
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    pair=pairs(a.data)[a.teacher]; write_json(out/'pair.json',pair)
    contract=dict(args=vars(a),pair=pair,report=sha(HERE/'report.md'),exclusions=sha(HERE/'pbc_excluded.json'),
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
    seed_all(a.seed)
    net=student(a.tiny,True,a.data).to(dev)
    if ck:
        hub.load_state_dict(ck['hub']); net.load_state_dict(ck['student'])
    for stage,total in [('hub',a.hub_steps),('student',a.steps)]:
        if ck and ck['stage']=='student' and stage=='hub': continue
        module=hub if stage=='hub' else net
        if stage=='student': hub.eval().requires_grad_(False)
        opt=torch.optim.AdamW([p for p in module.parameters() if p.requires_grad],lr=a.hub_lr if stage=='hub' else 1e-4,weight_decay=.05)
        sched=torch.optim.lr_scheduler.OneCycleLR(opt,a.hub_lr if stage=='hub' else 1e-4,total_steps=total,pct_start=.05 if total>=100 else .5)
        start=0
        gradlog=ck.get('gradlog',[]) if ck and ck['stage']==stage else []
        weights=ck.get('weights',[]) if ck and ck['stage']==stage else []
        ref=None
        if stage=='student' and a.arm=='H_matched':
            ref=TeacherReplay(a,d,dev)
            if ck and ck['stage']==stage: ref.load(ck['reference'])
        if ck and ck['stage']==stage:
            opt.load_state_dict(ck['opt'])
            for state in opt.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v): state[k]=v.to(dev)
            sched.load_state_dict(ck['sched']); start=ck['step']; restore(ck['rng'])
        def save(step):
            atomic(dict(contract=contract,stage=stage,step=step,hub=hub.state_dict(),student=net.state_dict(),
                        opt=opt.state_dict(),sched=sched.state_dict(),rng=rng(),gradlog=gradlog,weights=weights,
                        reference=ref.state() if ref else None),ckpath)
        batches=iter(student_loader(d,a,start)) if stage=='student' else None
        if stage=='student' and not (ck and ck['stage']==stage):seed_all(a.seed)
        stage_start=time.monotonic()
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
                raw,rr,xa,ya=next(batches);rr=rr.numpy()
                xg=legacy.pipeline((raw.to(dev).float()/d.scale).clamp(0,1).permute(0,3,1,2),True)
                xanchor=legacy.pipeline((xa.to(dev).float()/255.).permute(0,3,1,2),True)
                ya=ya.to(dev)
                zg=encode(net,xg)
                with torch.no_grad(): zl=hub.lf(d.x(rr,dev))
                dist=(zg-zl).square().sum(-1).mean()
                za=encode(net,xanchor)
                source_class=torch.as_tensor(d.gs_cls[d.t['hist_row'][rr]],device=dev)
                ce=F.cross_entropy(net['C'](za*16),ya)+F.cross_entropy(net['C'](zg*16),source_class)
                weight=.5
                if step<200 or (step+1)%100==0:
                    norms=gradient_norms(dict(latent=dist,anchor=ce),shared_blocks(net))
                    rec=dict(step=step+1,**norms)
                    if ref and step<200:
                        tnorm=ref.update(xg,xanchor,rr,ya)
                        target=.5*tnorm['anchor']/(tnorm['sig']+tnorm['dist']+.5*tnorm['anchor'])
                        weight=matched_weight(target,norms['latent'],norms['anchor'])
                        weights.append(weight)
                        rec.update(reference=tnorm,T_share=target)
                    elif ref:weight=float(np.mean(weights))
                    rec.update(anchor_weight=weight,H_share=weight*norms['anchor']/(norms['latent']+weight*norms['anchor']))
                    gradlog.append(rec)
                elif ref:weight=float(np.mean(weights))
                loss=dist+weight*ce
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(module.parameters(),1)
            opt.step(); sched.step()
            if stage=='student' and a.tiny:print(f'student update {step+1}',flush=True)
            if (step+1)%100==0 or step+1==total:
                print(json.dumps(dict(stage=stage,step=step+1,loss=float(loss.detach()),elapsed_s=round(time.monotonic()-stage_start,3))),flush=True); save(step+1)
            if STOP: save(step+1); raise SystemExit(143)
        if stage=='hub':
            atomic(dict(state=hub.state_dict(),mu=d.mu,sd=d.sd,contract=contract),out/'hub.pt')
            ck=None
    write_json(out/'gradient_norms.json',dict(shared='trainable ViT blocks',records=gradlog,
                 frozen_weight=float(np.mean(weights)) if weights else .5,calibration_steps=len(weights)))
    artifact=dict(hub=hub.state_dict(),student=net.state_dict(),scale=d.scale,names=d.names,
                  gs_train=d.gs_train,lf_train=d.lf_train,mu=d.mu,sd=d.sd,contract=contract,tiny=a.tiny)
    atomic(artifact,out/'hub_student.pt')
    write_json(out/'done.json',dict(done=True,checkpoint=str(out/'hub_student.pt'),tiny=a.tiny,contract=contract))
    if STOP: raise SystemExit(143)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[0,1],default=0); p.add_argument('--arm',choices=['H','H_matched'],default='H')
    p.add_argument('--workers',type=int,default=12);p.add_argument('--hub-lr',type=float,default=1e-3)
    p.add_argument('--manifest',action='store_true'); p.add_argument('--out',default=str(ROOT/'runs/n056v2_main/t0/H_s0_train'))
    p.add_argument('--data',default=str(ROOT)); p.add_argument('--teacher',type=int,choices=[0,1,2],default=0)
    p.add_argument('--device',default='cuda'); p.add_argument('--tiny',action='store_true')
    p.add_argument('--steps',type=int,default=3000); p.add_argument('--hub-steps',type=int,default=2500)
    p.add_argument('--bs',type=int,default=128); p.add_argument('--hub-bs',type=int,default=256)
    p.add_argument('--threads',type=int,default=16)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGUSR1,stop)
    if a.manifest:
        Path(a.out).mkdir(parents=True,exist_ok=True); write_json(Path(a.out)/'pairs.json',pairs(a.data)); return
    if a.tiny and a.device!='cpu': raise ValueError('tiny is a CPU engineering check')
    if not a.tiny and (a.steps,a.hub_steps,a.bs,a.hub_bs,a.hub_lr,a.workers)!=(3000,2500,128,256,1e-3,12):
        raise ValueError('production configuration differs from report.md')
    if a.arm=='H_matched' and a.seed!=0:raise ValueError('H_matched is preregistered seed0 only')
    train(a)
if __name__=='__main__': main()
