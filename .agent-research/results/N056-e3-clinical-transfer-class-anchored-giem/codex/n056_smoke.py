#!/usr/bin/env -S python3 -I -B
"""CPU engineering checks. Never predict on clinical evaluation images."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent

def sibling(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/codex_n056/smoke');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    tr=sibling('n056_hub_student');ev=sibling('n056_eval');ot=sibling('n056_ot_align')
    print('CPU',torch.__version__,'GPU available',torch.cuda.is_available(),flush=True)
    # Verify exact AST copies, including the complete augmentation function.
    original=ast.parse((tr.REPO/'scripts/train_gs_bind.py').read_text());local=ast.parse((HERE/'n056_legacy.py').read_text())
    for name in ['pipeline','ExtSet','macenko','_macenko_stains']:
        get=lambda tree:next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        assert ast.dump(get(original),include_attributes=False)==ast.dump(get(local),include_attributes=False)
    original=ast.parse((tr.REPO/'scripts/apply_gs_student.py').read_text());ns={}
    def assignment(name):return next(n for n in ast.walk(original) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets))
    ns['L'],ns['N']=['lymphocyte'],['neutrophil_segmented']
    for name in ['C','MC']:
        node=assignment(name);exec(compile(ast.Module(body=[node],type_ignores=[]),'<legacy>','exec'),ns)
        assert ns[name]==getattr(ev.contrasts,name)
    print('PASS verbatim augmentation, anchor loader and 10+15 contrasts',flush=True)
    pairs=tr.pairs();assert len(pairs)==3
    assert len(set(p['args']['lf_feats'] for p in pairs))==3
    assert len(set(p['args']['aux_feats'] for p in pairs))==1
    for p in pairs:assert Path(p['checkpoint']).exists()
    print('PASS 3 paired saved students; common v6c auxiliary teacher',flush=True)
    for domain in ['PBC','MLL23']:
        fit=ev.clinical_manifest(tr.ROOT,domain,'fit');test=ev.clinical_manifest(tr.ROOT,domain,'eval')
        assert not np.intersect1d(ev.record_ids(fit),ev.record_ids(test)).size
        print('PASS split',domain,'fit',len(fit),'eval',len(test),'(filenames only)',flush=True)
    # Exact known geometry: epsilon small recovers identity on two separated modes.
    x=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    m=ot.UOT().fit(x,x);z=m.transform(x)
    assert torch.allclose(z,x,atol=.01)
    state=m.state();copy=ot.UOT.load(state)
    assert torch.allclose(copy.transform(x,batch=1),z,atol=1e-6)
    assert torch.allclose(copy.transform(x.flip(0)),z.flip(0),atol=1e-6)
    assert torch.isfinite(m.transform(x*1000)).all()
    fixed=m.v.clone();m.transform(torch.randn(7,2));assert torch.equal(fixed,m.v)
    cond=ot.ConditionalUOT().fit(x.repeat(3,1),x.repeat(3,1),np.repeat(np.arange(3),4),np.repeat(np.arange(3),4))
    assert torch.allclose(cond.transform(x,torch.tensor([0,1,2,0])),x,atol=.01)
    print('PASS UOT identity, finite extremes, serialization, inductive/permutation invariance, conditional routing',m.diagnostics,flush=True)
    # Synthetic marker scores: every pre-registered contrast is evaluated.
    names=[str(n) for n in np.load(tr.ROOT/'targets_v4c.npz')['names']]
    for domain,cc in [('PBC',ev.contrasts.C),('MLL23',ev.contrasts.MC)]:
        classes=sorted(set(c for value in cc.values() for c in value[1]+value[2]));cl=np.repeat(classes,2)
        scores=np.zeros((len(cl),13));rows=ev.score_contrasts(scores,cl,names,domain)
        assert len(rows)==len(cc) and all(r['auroc']==.5 for r in rows)
        assert ev.score_contrasts(scores[:2],cl[:2],names,domain)[0]['auroc'] is None
    print('PASS all synthetic AUROCs=.5 and missing-class AUROCs=NA',flush=True)
    # Threshold boundary/tie and lack-of-expert-support checks.
    pp=np.tile(np.linspace(-2,2,120)[:,None],(1,13));sc=np.repeat(np.arange(3),40)
    pos=np.tile(np.r_[np.zeros(60),np.ones(60)][:,None],(1,8));st=np.repeat(np.arange(3),40)
    # enough measured observations per class after 3x tiling
    pos=np.tile(pos,(3,1));st=np.tile(st,3)
    ts=ev.thresholds(pp,sc,st,pos,names,np.arange(120))
    assert all(t['status']=='ok' for t in ts)
    missing=ev.thresholds(pp,np.full(120,-1),st,pos,names,np.arange(120));assert all(t['status']=='unavailable' for t in missing)
    result=ev.score_fractions(pp,np.array(['lymphocyte']*120),names,dict(primary=missing,proxy=ts))
    assert all(r['fraction'] is None for r in result if r['analysis']=='primary')
    assert all(r['status']=='proxy_class_and_mixture' for r in result if r['analysis']=='proxy')
    print('PASS primary unsupported fractions=NA; proxy fractions separately labeled',flush=True)
    # Source positivity uses only split-0 measured stained cells, never heldout values.
    rg=np.random.default_rng(7);vv=np.r_[rg.normal(-2,.15,150),rg.normal(2,.15,150),np.full(20,999.)]
    fake=dict(hist_row=np.full(320,-1),split=np.r_[np.zeros(300),np.full(20,2)],batch=np.zeros(320))
    base=dict(mask=np.ones((320,8),bool),T=np.tile(vv[:,None],(1,8)))
    positive,info=ev.positive_modes(fake,base)
    assert all(info[f'{j}/batch0']['status']=='pass' for j in range(8))
    assert np.all(positive[:150]==0) and np.all(positive[150:300]==1) and np.isnan(positive[300:]).all()
    print('PASS source mixture high-mode positivity and heldout-label exclusion',flush=True)
    # A complete artificial 3-teacher/6-arm table tests paired means, not clinical output.
    synth=out/'synthetic_summary'
    for t in range(3):
        for arm in ['T','H']:
            folder=synth/f't{t}'/arm;folder.mkdir(parents=True,exist_ok=True);metrics=[]
            for suffix in ['', '+OT','+CCOT']:
                for domain in ['PBC','MLL23']:
                    if suffix=='+CCOT' and domain=='MLL23':continue
                    cc=ev.contrasts.C if domain=='PBC' else ev.contrasts.MC
                    classes=sorted(set(c for v in cc.values() for c in v[1]+v[2]));cl=np.repeat(classes,2)
                    rows=ev.score_contrasts(np.zeros((len(cl),13)),cl,names,domain)
                    for r in rows:r.update(teacher=f't{t}',arm=arm+suffix,auroc=.5+.01*t+(.02 if arm=='H' else 0)+(.03 if suffix else 0))
                    metrics+=rows
            ev.csv_write(folder/'contrasts.csv',metrics)
            ev.csv_write(folder/'fractions.csv',[dict(analysis='primary',arm=arm,teacher=f't{t}',domain='PBC',clinical_class='lymphocyte',marker='single:CD3',expectation=.7,fraction=None,absolute_error=None)])
            ev.csv_write(folder/'sanity.csv',[dict(teacher=f't{t}',arm=arm,slide=0,n=4,autofl_r=.6)])
            tr.write_json(folder/'eval.done.json',dict(done=True))
    print('SYNTHETIC aggregate fixture follows (these are not clinical results)',flush=True)
    ev.aggregate(synth)
    pm=ev.readcsv(synth/'paired_macros.csv');assert abs(float(next(r['mean'] for r in pm if r['comparison']=='H minus T' and r['summary']=='all25'))-.02)<1e-8
    macros=ev.readcsv(synth/'macro_means.csv');assert all(r['mean']=='' for r in macros if r['arm'].endswith('CCOT') and r['summary']=='all25')
    print('PASS exactly-three-teacher aggregation and paired deltas; CCOT has no all25 score',flush=True)
    # Actual source images + permitted PBC A anchors through training. Two
    # executions compare uninterrupted and SIGTERM/resumed parameter equality.
    common=[sys.executable,'-I','-B',str(HERE/'n056_hub_student.py'),'--device','cpu','--tiny','--steps','4','--hub-steps','4','--bs','2','--hub-bs','8','--threads','2']
    fresh=out/'fresh';resume=out/'resumed'
    if fresh.exists() or resume.exists():raise ValueError('smoke output must be fresh for signal check')
    subprocess.run(common+['--out',str(fresh)],check=True)
    proc=subprocess.Popen(common+['--out',str(resume)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        print(line.rstrip(),flush=True)
        if line.startswith('ready hub'):
            proc.send_signal(signal.SIGTERM);break
    rest=proc.communicate(timeout=120)[0];print(rest,end='',flush=True)
    assert proc.returncode==143,proc.returncode
    checkpoint=torch.load(resume/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert all(k in checkpoint for k in ['opt','sched','rng','hub','student','step'])
    subprocess.run(common+['--out',str(resume)],check=True)
    x=torch.load(fresh/'hub_student.pt',weights_only=False,map_location='cpu');y=torch.load(resume/'hub_student.pt',weights_only=False,map_location='cpu')
    for section in ['hub','student']:
        assert all(torch.equal(x[section][k],y[section][k]) for k in x[section]),section
    print('PASS tiny real-source/anchor-A training, SIGTERM exit143, optimizer/RNG resume bitwise equal',flush=True)
    subprocess.run(common+['--out',str(resume)],check=True)
    print('PASS completed successor is idempotent',flush=True)
    # Real full T checkpoint loading and CPU inference on TWO BSCCM train cells.
    from types import SimpleNamespace
    opts=SimpleNamespace(device='cpu',arm='T',data=str(tr.ROOT))
    predictor=ev.Predictor(opts,pairs[0]);t=np.load(tr.ROOT/'targets_v4c.npz');rr=np.flatnonzero((t['hist_row']>=0)&(t['split']==0))[:2]
    hist=np.load(tr.ROOT/'pack/histology.npy',mmap_mode='r')
    batch=[np.clip(np.asarray(hist[t['hist_row'][i]],np.float32)/predictor.scale,0,1) for i in rr]
    p,z,zv=predictor.predict(batch)
    assert p.shape==(2,13) and z.shape==(2,256) and zv.shape==(2,8,256) and np.isfinite(p).all()
    print('PASS actual t0 T checkpoint, 8-view inference on 2 source training cells; no clinical test predictions',flush=True)
    print('ALL CPU SMOKE CHECKS PASSED',flush=True)
if __name__=='__main__':main()
