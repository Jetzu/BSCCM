#!/usr/bin/env -S python3 -I -B
"""Sealed source calibration, leakage-audited clinical inference and paired summaries."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import zipfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
import importlib.util
HERE=Path(__file__).resolve().parent

def sibling(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
tr=sibling('n056_hub_student');ot=sibling('n056_ot_align');contrasts=sibling('n056_contrasts')

EXPECT={'neutrophil':{'single:CD16':.95},'neutrophil_segmented':{'single:CD16':.95},
        'lymphocyte':{'single:CD3':.70},'monocyte':{'single:CD14':.95},'eosinophil':{'single:CD16':0.}}
CLASS={'lymphocyte':0,'neutrophil':1,'neutrophil_segmented':1,'monocyte':2,'eosinophil':1}

def check_stop():
    if tr.STOP: raise SystemExit(143)

def save_npz(path,**kw):
    path=Path(path);tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**kw);os.replace(tmp,path)

def csv_write(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def role(path,expected):
    f=np.load(path)
    if str(f['role'])!=expected:raise ValueError(f'{path}: expected role {expected}')
    return f

class Predictor:
    def __init__(self,a,pair):
        self.dev=torch.device(a.device);self.arm=a.arm
        self.names=[str(n) for n in np.load(Path(a.data)/'targets_v4c.npz')['names']]
        self.ckpt=pair['checkpoint'] if a.arm=='T' else str(Path(a.hub)/'hub_student.pt')
        if a.arm=='T':
            self.net=tr.student(False,False,a.data)
            self.net['S']=tr.legacy_mlp(256,len(self.names))
            state=torch.load(self.ckpt,map_location='cpu',weights_only=True)
            self.net.load_state_dict({k:v for k,v in state.items() if k.split('.')[0] in ['vit','PG','C','S']},strict=True)
            self.head=self.net['S'];self.scale=float(np.percentile(np.load(Path(a.data)/'pack/histology.npy',mmap_mode='r')[:500:5],99.9))
        else:
            ck=torch.load(self.ckpt,map_location='cpu',weights_only=False)
            if ck['tiny']:raise ValueError('tiny checkpoints cannot generate clinical results')
            if ck['contract']['pair']!=pair:raise ValueError('unpaired hub/student checkpoint')
            self.net=tr.student(False,False,a.data);self.net.load_state_dict(ck['student'])
            hub=tr.core.Spaces(256);hub.load_state_dict(ck['hub']);self.head=hub.H
            self.scale=ck['scale']
        self.net.to(self.dev).eval().requires_grad_(False);self.head.to(self.dev).eval().requires_grad_(False)
    @torch.no_grad()
    def decode(self,z):return self.head(z*16).float()
    @torch.no_grad()
    def predict(self,batch01):
        # Verbatim legacy interpolation/normalization and TTA order.
        x=torch.stack([F.interpolate(torch.tensor(np.ascontiguousarray(b),dtype=torch.float32).permute(2,0,1)[None],size=224,mode='bilinear',antialias=True)[0] for b in batch01]).to(self.dev)
        mean=torch.tensor([.485,.456,.406],device=self.dev).view(1,3,1,1)
        std=torch.tensor([.229,.224,.225],device=self.dev).view(1,3,1,1)
        x=(x.clamp(0,1)-mean)/std;zs=[];ps=[]
        for k,f in [(k,f) for k in range(4) for f in (0,1)]:
            xv=torch.rot90(x,k,(2,3));xv=xv.flip(3) if f else xv
            z=tr.encode(self.net,xv);zs.append(z);ps.append(self.decode(z))
        zv=torch.stack(zs,1)
        return torch.stack(ps).mean(0).cpu().numpy(),F.normalize(zv.mean(1),dim=1).cpu().numpy(),zv.cpu().numpy()
    @torch.no_grad()
    def aligned(self,zviews,mapper):
        p=[]
        for s in range(0,len(zviews),128):
            check_stop();z=torch.as_tensor(zviews[s:s+128],device=self.dev);n,v,d=z.shape
            flat=z.reshape(-1,d)
            if isinstance(mapper,ot.ConditionalUOT):
                # Frozen C routes each view. Evaluation labels never reach the map.
                routes=self.net['C'](flat*16).argmax(1);mapped=mapper.transform(flat,routes)
            else:mapped=mapper.transform(flat)
            p.append(self.decode(mapped).reshape(n,v,-1).mean(1).cpu().numpy())
        return np.concatenate(p)

def sorted_names(z,ext):
    return sorted(n for n in z.namelist() if n.lower().endswith(ext) and '__MACOSX' not in n and not os.path.basename(n).startswith('.'))

def clinical_manifest(data,domain,which):
    if which not in ['fit','eval']:raise ValueError(which)
    records=[]
    if domain=='PBC':
        path=Path(data)/'external/pbc/PBC_dataset_normal_DIB.zip'
        with zipfile.ZipFile(path) as z:names=sorted_names(z,('.jpg',))
        for n in names[0::2] if which=='fit' else names[1::2]:records.append((str(path),n,n.split('/')[-2]))
    elif domain=='MLL23':
        for path in sorted((Path(data)/'external/mll23').glob('*.zip')):
            with zipfile.ZipFile(path) as z:names=sorted_names(z,('.jpg','.png','.tif','.tiff'))
            selected=[names[i] for i in np.random.default_rng(0).permutation(len(names))[:1500]]
            for n in selected[0::2] if which=='fit' else selected[1::2]:records.append((str(path),n,path.stem))
    else:raise ValueError(domain)
    if not records:raise ValueError('empty clinical manifest')
    return records

def record_ids(records):return np.array([p+'::'+n for p,n,c in records])

def cache_predict(path,predictor,ids,load_image,bs,role_name,classes=None):
    path=Path(path)
    signature=tr.sha(predictor.ckpt)
    if path.exists():
        old=role(path,role_name)
        if not np.array_equal(old['ids'],ids) or str(old['model_sha'])!=signature:raise ValueError('stale inference cache')
        return old
    # A completed image batch is an atomic cursor; SIGTERM can resume inference.
    parts=path.with_suffix('.parts');parts.mkdir(exist_ok=True)
    manifest=parts/'manifest.json'
    contract=dict(ids_hash=__import__('hashlib').sha256(np.asarray(ids).tobytes()).hexdigest(),model_sha=signature,bs=bs,role=role_name)
    if manifest.exists() and json.loads(manifest.read_text())!=contract:raise ValueError('changed inference cursor')
    tr.write_json(manifest,contract)
    pp=[];zz=[];vv=[]
    for start in range(0,len(ids),bs):
        check_stop();part=parts/f'{start:07d}.npz'
        if not part.exists():
            p,z,v=predictor.predict([load_image(i) for i in range(start,min(start+bs,len(ids)))])
            save_npz(part,pred=p,z=z,zviews=v)
        with np.load(part) as f:pp.append(f['pred']);zz.append(f['z']);vv.append(f['zviews'])
    kw=dict(pred=np.concatenate(pp),z=np.concatenate(zz),zviews=np.concatenate(vv),ids=ids,
            names=np.array(predictor.names),role=np.array(role_name),model_sha=np.array(signature))
    if classes is not None:kw['cls']=classes
    save_npz(path,**kw);return np.load(path)

def positive_modes(t,base):
    h,sp,b=t['hist_row'],t['split'],t['batch'];positive=np.full((len(h),8),np.nan,np.float32);info={}
    for j in range(8):
        for batch in [0,1]:
            rr=np.flatnonzero((h<0)&(sp==0)&(b==batch)&base['mask'][:,j])
            values=base['T'][rr,j].astype(np.float64);values=values[np.isfinite(values)]
            key=f'{j}/batch{batch}'
            if len(values)<100:info[key]=dict(status='insufficient source observations');continue
            g=GaussianMixture(2,random_state=56,n_init=3,reg_covar=1e-5).fit(values[:,None])
            means=g.means_[:,0];sd=np.sqrt(g.covariances_[:,0,0]);sep=float(abs(means[1]-means[0])/np.sqrt(np.mean(sd**2)))
            info[key]=dict(means=means.tolist(),sd=sd.tolist(),weights=g.weights_.tolist(),separation=sep,training_rows=rr.tolist())
            if g.weights_.min()<.02 or sep<1:info[key]['status']='failed fixed mixture QC';continue
            positive[rr,j]=(g.predict_proba(base['T'][rr,j,None])[:,means.argmax()]>=.5).astype(np.float32)
            info[key]['status']='pass'
    return positive,info

def thresholds(pred,source_cls,stained_cls,positive,names,gs_train):
    out=[]
    for c in range(3):
        src=source_cls==c
        for j in range(8):
            r=(stained_cls==c)&np.isfinite(positive[:,j]);n=int(r.sum())
            row=dict(class_index=c,marker=names[j],n_stained=n,n_gs=int(src.sum()),status='unavailable',threshold=None,rate=None)
            if n>=100 and src.sum()>=30:
                rate=float(positive[r,j].mean());q=float(np.quantile(pred[src,j],1-rate)) if 0<rate<1 else None
                # Infinity is serialized by an explicit boundary rule, not JSON NaN.
                row.update(status='ok',rate=rate,threshold=q,boundary='all' if rate==1 else 'none' if rate==0 else 'finite')
                got=float((pred[src,j]>q).mean()) if q is not None else rate
                row['training_tie_error']=abs(got-rate)
            out.append(row)
    return out

def prepare(a,pair,pred):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    seal=out/'seal.json'
    if seal.exists():verify_seal(out,pred,pair);return
    # Guard against accidentally calibrating after a clinical prediction exists.
    if list(out.glob('*_fit.npz')) or list(out.glob('*_eval.npz')):raise ValueError('clinical predictions preceded calibration seal')
    d=tr.Data(a,pair);h=d.t['hist_row'];ids=d.gs_train
    source=cache_predict(out/'source.npz',pred,ids,lambda i:np.clip(np.asarray(d.hist[h[ids[i]]],np.float32)/pred.scale,0,1),a.bs,'source_train',d.gs_cls[h[ids]].astype(str))
    with np.load(Path(a.data)/'targets_v3c.npz') as f:base={k:f[k].copy() for k in ['T','mask','hist_row','split','batch','names']}
    for key in ['hist_row','split','batch','names']:assert np.array_equal(base[key],d.t[key])
    positive,modes=positive_modes(d.t,base)
    # Literal measured-class support. Missing GS expert classes also forbids it.
    true_gs=d.t['cls3'][ids]
    primary=thresholds(source['pred'],true_gs,d.t['cls3'],positive,d.names,ids)
    # Exploratory source-only one-hot LF ridge classifier.
    rr=np.flatnonzero((h<0)&(d.t['split']==0)&(d.t['cls3']>=0))
    x=d.x(rr,pred.dev)[:,:512];xb=torch.cat([x,torch.ones(len(x),1,device=pred.dev)],1)
    y=F.one_hot(torch.as_tensor(d.t['cls3'][rr],device=pred.dev).long(),3).float()
    reg=10*torch.eye(xb.shape[1],device=pred.dev);reg[-1,-1]=0
    weight=torch.linalg.solve(xb.T@xb+reg,xb.T@y)
    pseudo=np.full(len(h),-1,np.int64);rows=np.flatnonzero((h<0)&(d.t['split']==0))
    for s in range(0,len(rows),4096):
        check_stop();r=rows[s:s+4096];x=d.x(r,pred.dev)[:,:512]
        pseudo[r]=(x@weight[:-1]+weight[-1]).argmax(1).cpu().numpy()
    proxy=thresholds(source['pred'],d.gs_cls[h[ids]],pseudo,positive,d.names,ids)
    tr.atomic(dict(weight=weight.cpu(),training_rows=rr,source_rows=rows,source_pseudo_class=pseudo[rows]),out/'proxy_classifier.pt')
    save_npz(out/'calibration_rows.npz',gs_train=ids,source_class=d.gs_cls[h[ids]],measured_positive=positive,
             classifier_train_rows=rr,stained_rows=rows,stained_pseudo_class=pseudo[rows])
    tr.write_json(out/'thresholds.json',dict(primary=primary,proxy=proxy,modes=modes,expectations=EXPECT))
    # Freeze protocol + thresholds + models BEFORE clinical image inference.
    tr.write_json(seal,dict(report_sha=tr.sha(HERE/'report.md'),threshold_sha=tr.sha(out/'thresholds.json'),
                source_sha=tr.sha(out/'source.npz'),model_sha=tr.sha(pred.ckpt),pair=pair,
                code={p.name:tr.sha(p) for p in HERE.glob('n056_*.py')}))
    print('source calibration sealed; clinical inference now permitted',flush=True)

def verify_seal(out,pred,pair):
    seal=json.loads((Path(out)/'seal.json').read_text())
    checks=dict(report_sha=tr.sha(HERE/'report.md'),threshold_sha=tr.sha(Path(out)/'thresholds.json'),
                source_sha=tr.sha(Path(out)/'source.npz'),model_sha=tr.sha(pred.ckpt),pair=pair,
                code={p.name:tr.sha(p) for p in HERE.glob('n056_*.py')})
    if seal!=checks:raise ValueError('calibration/model/preregistration seal changed')

def clinical_cache(a,pred,domain,which):
    records=clinical_manifest(a.data,domain,which);other=clinical_manifest(a.data,domain,'eval' if which=='fit' else 'fit')
    ids=record_ids(records)
    if np.intersect1d(ids,record_ids(other)).size:raise ValueError('fit/eval image overlap')
    handles={}
    def image(i):
        p,n,c=records[i]
        if p not in handles:handles[p]=zipfile.ZipFile(p)
        return np.asarray(Image.open(io.BytesIO(handles[p].read(n))).convert('RGB'),np.float32)/255.
    try:return cache_predict(Path(a.out)/f'{domain}_{which}.npz',pred,ids,image,a.bs,which,np.array([r[2] for r in records]))
    finally:
        for z in handles.values():z.close()

def fit_maps(a,pred,domain):
    out=Path(a.out);path=out/f'{domain}_ot.pt'
    if path.exists():return
    source=role(out/'source.npz','source_train');fit=clinical_cache(a,pred,domain,'fit')
    model=ot.UOT().fit(torch.as_tensor(fit['z'],device=pred.dev),torch.as_tensor(source['z'],device=pred.dev))
    state=dict(unconditional=model.state(),fit_ids=fit['ids'],source_ids=source['ids'])
    if domain=='PBC':
        labels=fit['cls'];keep=np.isin(labels,tr.ANCH)
        cond=ot.ConditionalUOT().fit(torch.as_tensor(fit['z'][keep],device=pred.dev),torch.as_tensor(source['z'],device=pred.dev),
                                   np.array([tr.ANCH.index(c) for c in labels[keep]]),source['cls'].astype(int))
        state['conditional']=cond.state();state['conditional_fit_ids']=fit['ids'][keep]
    tr.atomic(state,path)
    print(domain,json.dumps(model.diagnostics),flush=True)

def stratum(domain,key):
    if domain=='PBC':
        if key=='CD123 basophil>rest':return 'failing'
        if key=='CD16 neutrophil>eosinophil':return 'teacher_dependent'
        return 'robust'
    if key in ['segneut>lym CD16','mono>lym+neut CD14','lym>neut CD3','hairy>lym HLA-DR','myelocyte<segneut CD16']:return 'robust'
    if key in ['LGL>lym CD56','LGL>lym CD16','baso>rest CD123','plasma<lym CD45','normoblast<lym CD45','hairy>lym CD19','hairy<lym CD3']:return 'failing'
    if key=='eos<segneut CD16':return 'teacher_dependent'
    return 'other'

def score_contrasts(pred,classes,names,domain):
    cc=contrasts.C if domain=='PBC' else contrasts.MC;rows=[]
    for key,value in cc.items():
        marker,pos,neg=value[:3];sign=value[3] if len(value)==4 else 1
        selected=np.isin(classes,pos+neg);y=np.isin(classes[selected],pos)
        val=float(roc_auc_score(y,sign*pred[selected,names.index(marker)])) if len(np.unique(y))==2 else None
        rows.append(dict(domain=domain,contrast=key,marker=marker,stratum=stratum(domain,key),auroc=val,n=int(selected.sum()),fine=domain=='MLL23' and key not in ['segneut>lym CD16','mono>lym+neut CD14','lym>neut CD3']))
    return rows

def score_fractions(pred,classes,names,calibration):
    rows=[]
    for analysis in ['primary','proxy']:
        lookup={(x['class_index'],x['marker']):x for x in calibration[analysis]}
        for c in sorted(set(classes)):
            for marker in names[:8]:
                t=lookup.get((CLASS.get(c,-1),marker));expected=EXPECT.get(c,{}).get(marker)
                row=dict(analysis=analysis,clinical_class=c,marker=marker,n=int((classes==c).sum()),fraction=None,
                         expectation=expected,absolute_error=None,status='unavailable')
                if t and t['status']=='ok':
                    scores=pred[classes==c,names.index(marker)]
                    frac=1. if t['boundary']=='all' else 0. if t['boundary']=='none' else float((scores>t['threshold']).mean())
                    row.update(fraction=frac,absolute_error=abs(frac-expected) if expected is not None else None,
                               status='proxy_class_and_mixture' if analysis=='proxy' else 'measured_class')
                rows.append(row)
    return rows

def evaluate(a,pair,pred):
    verify_seal(a.out,pred,pair);out=Path(a.out);cal=json.loads((out/'thresholds.json').read_text());metrics=[];fractions=[]
    for domain in ['PBC','MLL23']:
        fit_maps(a,pred,domain);check_stop();ev=clinical_cache(a,pred,domain,'eval')
        state=torch.load(out/f'{domain}_ot.pt',map_location='cpu',weights_only=False)
        if np.intersect1d(state['fit_ids'],ev['ids']).size:raise ValueError('OT eval overlap')
        maps={'raw':None,'OT':ot.UOT.load(state['unconditional'],pred.dev)}
        if 'conditional' in state:maps['CCOT']=ot.ConditionalUOT.load(state['conditional'],pred.dev)
        for suffix,mapper in maps.items():
            arm=a.arm if mapper is None else a.arm+'+'+suffix
            p=ev['pred'] if mapper is None else pred.aligned(ev['zviews'],mapper)
            rows=score_contrasts(p,ev['cls'],pred.names,domain)
            for r in rows:r.update(teacher=f't{a.teacher}',arm=arm)
            metrics+=rows
            rows=score_fractions(p,ev['cls'],pred.names,cal)
            for r in rows:r.update(teacher=f't{a.teacher}',arm=arm,domain=domain)
            fractions+=rows
            save_npz(out/f'{domain}_{arm}_pred.npz',pred=p,cls=ev['cls'],ids=ev['ids'],names=np.array(pred.names),role='eval')
    csv_write(out/'contrasts.csv',metrics);csv_write(out/'fractions.csv',fractions)
    sanity(a,pred)
    tr.write_json(out/'eval.done.json',dict(done=True,teacher=f't{a.teacher}',arm=a.arm))

def sanity(a,pred):
    t=np.load(Path(a.data)/'targets_v6c.npz');h,sp=t['hist_row'],t['split'];ids=np.flatnonzero((h>=0)&(sp==2));hist=np.load(Path(a.data)/'pack/histology.npy',mmap_mode='r')
    f=cache_predict(Path(a.out)/'sanity.npz',pred,ids,lambda i:np.clip(np.asarray(hist[h[ids[i]]],np.float32)/pred.scale,0,1),a.bs,'source_eval')
    rows=[]
    for slide in np.unique(t['slide'][ids]):
        v=(t['slide'][ids]==slide)&t['mask'][ids,-1];p=f['pred'][v,-1];y=t['Y'][ids[v],-1]
        r=float(np.corrcoef(p,y)[0,1]) if len(p)>2 and p.std()>0 and y.std()>0 else None
        rows.append(dict(teacher=f't{a.teacher}',arm=a.arm,slide=int(slide),n=int(v.sum()),autofl_r=r))
    csv_write(Path(a.out)/'sanity.csv',rows)

def readcsv(path):
    with open(path) as f:return list(csv.DictReader(f))

def aggregate(root):
    root=Path(root);allrows=[];fr=[];san=[]
    for t in range(3):
        for arm in ['T','H']:
            d=root/f't{t}'/arm
            if not (d/'eval.done.json').exists():raise ValueError(f'missing complete paired run {d}')
            allrows+=readcsv(d/'contrasts.csv');fr+=readcsv(d/'fractions.csv');san+=readcsv(d/'sanity.csv')
    csv_write(root/'contrasts_all.csv',allrows);csv_write(root/'fractions_all.csv',fr);csv_write(root/'sanity_all.csv',san)
    def num(s):return float(s) if s not in ['',None] else None
    def stats(v):
        return dict(n_teachers=len(v),mean=float(np.mean(v)) if len(v)==3 else None,sd=float(np.std(v,ddof=1)) if len(v)==3 else None)
    sanity_means=[]
    for arm,slide in sorted(set((r['arm'],r['slide']) for r in san)):
        rr=[r for r in san if r['arm']==arm and r['slide']==slide]
        assert len({r['teacher'] for r in rr})==3
        sanity_means.append(dict(arm=arm,slide=slide,**stats([num(r['autofl_r']) for r in rr if num(r['autofl_r']) is not None])))
    csv_write(root/'sanity_means.csv',sanity_means)
    summary=[]
    for arm,domain,key in sorted(set((r['arm'],r['domain'],r['contrast']) for r in allrows)):
        rs=[r for r in allrows if (r['arm'],r['domain'],r['contrast'])==(arm,domain,key)]
        assert len({r['teacher'] for r in rs})==3
        summary.append(dict(arm=arm,domain=domain,contrast=key,stratum=rs[0]['stratum'],**stats([num(r['auroc']) for r in rs if num(r['auroc']) is not None])))
    csv_write(root/'contrast_means.csv',summary)
    macros=[]
    for arm in sorted(set(r['arm'] for r in allrows)):
        subsets={'all25':lambda r:True,'PBC10':lambda r:r['domain']=='PBC','MLL23_15':lambda r:r['domain']=='MLL23',
                 'MLL23_fine12':lambda r:r['domain']=='MLL23' and r['fine']=='True'}
        for dom in ['PBC','MLL23']:
            for group in ['robust','failing','teacher_dependent','other']:
                subsets[dom+'_'+group]=lambda r,d=dom,g=group:r['domain']==d and r['stratum']==g
        for label,test in subsets.items():
            # Conditional OT is intentionally PBC-only; never label 10 scores all25.
            vals=[]
            for teacher in ['t0','t1','t2']:
                rr=[r for r in allrows if r['arm']==arm and r['teacher']==teacher and test(r)]
                required={'all25':25,'PBC10':10,'MLL23_15':15,'MLL23_fine12':12}.get(label)
                if not rr or (required is not None and len(rr)!=required) or any(num(r['auroc']) is None for r in rr):continue
                vals.append(float(np.mean([num(r['auroc']) for r in rr])))
            macros.append(dict(arm=arm,summary=label,**stats(vals)))
    csv_write(root/'macro_means.csv',macros)
    paired=[]
    for new,base in [('H','T'),('T+OT','T'),('H+OT','H'),('H+OT','T+OT'),('T+CCOT','T'),('H+CCOT','H')]:
        for domain,key in sorted(set((r['domain'],r['contrast']) for r in allrows)):
            delta=[]
            for t in ['t0','t1','t2']:
                matches={r['arm']:num(r['auroc']) for r in allrows if r['teacher']==t and r['domain']==domain and r['contrast']==key and r['arm'] in [new,base]}
                if all(k in matches and matches[k] is not None for k in [new,base]):delta.append(matches[new]-matches[base])
            paired.append(dict(comparison=new+' minus '+base,domain=domain,contrast=key,**stats(delta)))
    csv_write(root/'paired_contrasts.csv',paired)
    # Macro deltas must be computed within teacher, then averaged.
    pm=[]
    for new,base in [('H','T'),('T+OT','T'),('H+OT','H'),('H+OT','T+OT')]:
        for label,domain,count in [('all25',None,25),('PBC10','PBC',10),('MLL23_15','MLL23',15)]:
            vals=[]
            for t in ['t0','t1','t2']:
                # Recompute from original records; paired means are not teacher observations.
                aa=[r for r in allrows if r['teacher']==t and r['arm']==new and (domain is None or r['domain']==domain)]
                bb=[r for r in allrows if r['teacher']==t and r['arm']==base and (domain is None or r['domain']==domain)]
                if len(aa)==len(bb)==count and all(num(r['auroc']) is not None for r in aa+bb):vals.append(np.mean([num(r['auroc']) for r in aa])-np.mean([num(r['auroc']) for r in bb]))
            pm.append(dict(comparison=new+' minus '+base,summary=label,**stats(vals)))
    csv_write(root/'paired_macros.csv',pm)
    fq=[]
    for key in sorted(set((r['analysis'],r['arm'],r['domain'],r['clinical_class'],r['marker']) for r in fr)):
        rr=[r for r in fr if tuple(r[k] for k in ['analysis','arm','domain','clinical_class','marker'])==key]
        fq.append(dict(zip(['analysis','arm','domain','clinical_class','marker'],key))|dict(expectation=rr[0]['expectation'],
                  **stats([num(r['fraction']) for r in rr if num(r['fraction']) is not None]),
                  mae=float(np.mean([num(r['absolute_error']) for r in rr])) if len(rr)==3 and all(num(r['absolute_error']) is not None for r in rr) else None))
    csv_write(root/'fraction_means.csv',fq)
    # Write results separately: never mutate the sealed report.md.
    best=[r for r in macros if r['summary']=='all25' and r['mean'] is not None]
    winner=max(best,key=lambda r:r['mean']) if best else None
    tr.write_json(root/'results.json',dict(best_macro=winner,macro_means=macros,paired_macro=pm,
                    caveat='RQ3 literal measured-class fractions unavailable; proxy MAE is exploratory.'))
    print(json.dumps(dict(best_macro=winner),indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','fit','evaluate','all','aggregate'],default='all')
    p.add_argument('--data',default=str(tr.ROOT));p.add_argument('--out',required=True)
    p.add_argument('--teacher',type=int,choices=[0,1,2],default=0);p.add_argument('--arm',choices=['T','H'],default='T')
    p.add_argument('--hub',default='');p.add_argument('--device',default='cuda');p.add_argument('--bs',type=int,default=64)
    p.add_argument('--threads',type=int,default=16)
    a=p.parse_args();a.tiny=False;torch.set_num_threads(a.threads)
    signal.signal(signal.SIGTERM,tr.stop);signal.signal(signal.SIGUSR1,tr.stop)
    if a.stage=='aggregate':aggregate(a.out);return
    pair=tr.pairs(a.data)[a.teacher]
    done=Path(a.out)/'eval.done.json'
    if a.stage in ['all','evaluate'] and done.exists():
        seal=json.loads((Path(a.out)/'seal.json').read_text())
        model=pair['checkpoint'] if a.arm=='T' else str(Path(a.hub)/'hub_student.pt')
        if seal['report_sha']!=tr.sha(HERE/'report.md') or seal['model_sha']!=tr.sha(model) or seal['pair']!=pair or seal['code']!={p.name:tr.sha(p) for p in HERE.glob('n056_*.py')}:
            raise ValueError('completed evaluation contract changed')
        print('completed evaluation: immediate exit',flush=True);return
    pred=Predictor(a,pair)
    if a.stage in ['prepare','all']:prepare(a,pair,pred)
    verify_seal(a.out,pred,pair)
    if a.stage=='fit':
        for domain in ['PBC','MLL23']:fit_maps(a,pred,domain)
    if a.stage in ['evaluate','all']:evaluate(a,pair,pred)
if __name__=='__main__':main()
