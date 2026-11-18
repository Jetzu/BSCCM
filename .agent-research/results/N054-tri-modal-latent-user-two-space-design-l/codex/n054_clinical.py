#!/usr/bin/env python3
"""Apply a frozen N054 arm to PBC/MLL23. Resumes inference from <out>/ckpt.pt.
Contrast definitions and RGB resize/normalization/TTA are copied from
scripts/apply_gs_student.py; no clinical labels enter training or selection.
"""
import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score


def sibling(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod


n=sibling('n054_trimodal'); contrasts=sibling('n054_contrasts')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--arm',choices=['A0','A1','A2','A3'],required=True)
    ap.add_argument('--holdout',choices=['batch0','batch1'],default='batch1')
    ap.add_argument('--data',default=n.ROOT); ap.add_argument('--baseline',default='')
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--tta',type=int,choices=[0,1],default=1); ap.add_argument('--pbc-half',choices=['all','A','B'],default='all')
    ap.add_argument('--mll-max',type=int,default=1500); ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--smoke',action='store_true'); cli=ap.parse_args()
    out=Path(cli.out); out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(cli.threads); n.seed_all(0)
    signal.signal(signal.SIGTERM,n.stop_handler); signal.signal(signal.SIGUSR1,n.stop_handler)
    args=n.parser().parse_args(['eval','--arm',cli.arm,'--holdout',cli.holdout,'--out',cli.run,'--data',cli.data])
    if cli.arm!='A0':
        ck=torch.load(Path(cli.run)/'gs_encoder.pt',map_location='cpu',weights_only=False)
        args=argparse.Namespace(**ck['args']); assert args.arm==cli.arm and args.holdout==cli.holdout
        assert args.smoke==cli.smoke,'--smoke must match the trained checkpoint'
    args.smoke=cli.smoke; args.baseline=cli.baseline; args.data=cli.data
    dev=torch.device('cuda' if cli.device=='auto' and torch.cuda.is_available() else 'cpu' if cli.device=='auto' else cli.device)
    data=n.Data(args); pred=n.Predictor(args,data,dev)
    MEAN=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
    STD=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    @torch.no_grad()
    def predict(batch01):
        # Verbatim external preprocessing from apply_gs_student.py (224 resize,
        # ImageNet normalization, rot90 x flip). Smoke backbone then uses 32px.
        x=torch.stack([F.interpolate(torch.tensor(np.ascontiguousarray(b),dtype=torch.float32).permute(2,0,1)[None],size=224,mode='bilinear',antialias=True)[0] for b in batch01]).to(dev)
        x=(x.clamp(0,1)-MEAN)/STD
        if cli.smoke and cli.arm!='A0': x=F.interpolate(x,size=32,mode='bilinear',antialias=True)
        P,Z=0,0; views=[(k,f) for k in range(4) for f in (0,1)] if cli.tta else [(0,0)]
        for k,f in views:
            xv=torch.rot90(x,k,(2,3)); xv=xv.flip(3) if f else xv
            pp,zz,_=pred.images(xv); P=P+pp; Z=Z+zz
        z=Z/np.linalg.norm(Z,axis=1,keepdims=True).clip(min=1e-12)
        return P/len(views),z
    # Inference checkpoint is bound to exact model artifact paths + mtimes/sizes.
    model_files=[Path(cli.run)/f for f in ('lf_spaces.pt','gs_encoder.pt','flow.pt')] if cli.arm!='A0' else [Path(cli.baseline or Path(cli.data)/f'runs/n029_tt6c_x_{cli.holdout}/gs_bind.pt')]
    fingerprint=[(str(f.resolve()),f.stat().st_size,f.stat().st_mtime_ns) for f in model_files if f.exists()]
    config=dict(cli=vars(cli),model_files=fingerprint)
    ckpath=out/'ckpt.pt'; state=dict(config=config,archives={})
    if ckpath.exists():
        state=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert state['config']==config,'clinical checkpoint config/model differs'
        print('resume clinical inference',flush=True)
    def infer_archive(path,names,key,classes):
        zf=zipfile.ZipFile(path); rec=state['archives'].setdefault(key,dict(names=names,done=0,p=[],z=[]))
        assert rec['names']==names
        for s in range(rec['done'],len(names),cli.batch_size):
            batch=[np.asarray(Image.open(io.BytesIO(zf.read(fn))).convert('RGB'),np.float32)/255. for fn in names[s:s+cli.batch_size]]
            pp,zz=predict(batch); rec['p'].append(pp); rec['z'].append(zz); rec['done']=s+len(batch)
            n.atomic_save(state,ckpath)
            if n.STOP: raise SystemExit(99)
        zf.close(); pp=np.concatenate(rec['p']); zz=np.concatenate(rec['z'])
        return pp,zz,np.asarray(classes),np.asarray(names)
    def members(path):
        with zipfile.ZipFile(path) as zf:
            return sorted(fn for fn in zf.namelist() if fn.lower().endswith(('.jpg','.png','.tif','.tiff')) and '__MACOSX' not in fn and not os.path.basename(fn).startswith('.'))
    pbc=Path(cli.data)/'external/pbc/PBC_dataset_normal_DIB.zip'
    # PBC original filter is .jpg only.
    fn=[f for f in members(pbc) if f.lower().endswith('.jpg')]
    fn=fn if cli.pbc_half=='all' else fn[0::2] if cli.pbc_half=='A' else fn[1::2]
    if cli.smoke:
        fn=[f for c in sorted({f.split('/')[-2] for f in fn}) for f in [x for x in fn if x.split('/')[-2]==c][:2]]
    cls=[f.split('/')[-2] for f in fn]
    pp,zp,cp,fp=infer_archive(pbc,fn,'pbc',cls)
    np.savez(out/'pbc_gs_student.npz',pred=pp,z=zp,cls=cp,files=fp,names=data.names)
    mll=[]; zall=[]; cm=[]; fm=[]
    for path in sorted((Path(cli.data)/'external/mll23').glob('*.zip')):
        fn=members(path); limit=2 if cli.smoke else cli.mll_max
        fn=[fn[i] for i in np.random.default_rng(0).permutation(len(fn))[:limit]]
        if not fn: continue
        p,z,c,f=infer_archive(path,fn,'mll_'+path.stem,[path.stem]*len(fn)); mll.append(p); zall.append(z); cm.extend(c); fm.extend([path.name+'/'+s for s in f])
        print('MLL23',path.stem,len(fn),flush=True)
    if not mll: raise ValueError('no MLL23 archives')
    pm=np.concatenate(mll); cm=np.asarray(cm)
    np.savez(out/'mll23_gs_student.npz',pred=pm,z=np.concatenate(zall),cls=cm,files=np.asarray(fm),names=data.names)
    rows=[]
    for dataset,table,p,c in [('PBC',contrasts.C,pp,cp),('MLL23',contrasts.MC,pm,cm)]:
        for contrast,spec in table.items():
            marker,pos,neg,*sgn=spec; sign_=sgn[0] if sgn else 1
            mask=np.isin(c,pos+neg); y=np.isin(c[mask],pos)
            auc=float(roc_auc_score(y,sign_*p[mask,data.names.index(marker)])) if len(np.unique(y))==2 else None
            rows.append(dict(dataset=dataset,contrast=contrast,marker=marker,auroc=auc,n_pos=int(y.sum()),n_neg=int((~y).sum()),smoke=cli.smoke))
    means={dataset:{c:{name:float(p[classes==c,k].mean()) for k,name in enumerate(data.names)} for c in sorted(set(classes))}
           for dataset,p,classes in [('PBC',pp,cp),('MLL23',pm,cm)]}
    n.csv_write(out/'auroc.csv',rows); n.json_write(out/'clinical.json',dict(contrasts=rows,class_means=means,config=config))
    print(json.dumps(dict(status='done',pbc_n=len(pp),mll_n=len(pm),contrasts=len(rows),smoke=cli.smoke)),flush=True)


if __name__=='__main__': main()
