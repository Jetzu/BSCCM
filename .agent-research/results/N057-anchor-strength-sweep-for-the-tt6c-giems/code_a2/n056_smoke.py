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
    p=argparse.ArgumentParser();p.add_argument('--out',default='/tmp/codex_n056v2/smoke');a=p.parse_args()
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
    allpairs=[p for seed in range(3) for p in tr.pairs(seed=seed) if p is not None]
    assert len(allpairs)==7 and all(Path(p['checkpoint']).exists() for p in allpairs)
    tr.write_json(HERE/'teachers.json',allpairs)
    print('PASS all 7 saved T students; H seeds0/1, H_matched seed0; common v6c auxiliary teacher',flush=True)
    for domain in ['PBC','MLL23']:
        fit=ev.clinical_manifest(tr.ROOT,domain,'fit');test=ev.clinical_manifest(tr.ROOT,domain,'eval')
        assert not np.intersect1d(ev.record_ids(fit),ev.record_ids(test)).size
        if domain=='PBC':
            drop=ev.duplicates.excluded(tr.ROOT/'external/pbc/PBC_dataset_normal_DIB.zip')
            assert not any(n in drop for p,n,c in fit+test)
            assert any('EO_225902' in n for n in drop) and any('BNE_191112' in n for n in drop)
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
    # Full seeded grid, unequal T seed counts, and both sides of noise-floor rule.
    synth=out/'synthetic_summary'
    for t,arm,seed,folder in ev.run_grid(synth):
        folder.mkdir(parents=True,exist_ok=True);metrics=[]
        for suffix in ['', '+OT_tau0.91','+OT_tau0.30','+mean','+CORAL','+kNN','+CCOT']:
            for domain in ['PBC','MLL23']:
                if suffix=='+CCOT' and domain=='MLL23':continue
                cc=ev.contrasts.C if domain=='PBC' else ev.contrasts.MC
                classes=sorted(set(c for v in cc.values() for c in v[1]+v[2]));cl=np.repeat(classes,2)
                rows=ev.score_contrasts(np.zeros((len(cl),13)),cl,names,domain)
                for r in rows:r.update(teacher=f't{t}',seed=seed,arm=arm+suffix,auroc=.5+.01*t+.01*seed+(.04 if arm=='H' else 0))
                metrics+=rows
        ev.csv_write(folder/'contrasts.csv',metrics)
        ev.csv_write(folder/'fractions.csv',[dict(analysis='primary',arm=arm,teacher=f't{t}',seed=seed,domain='PBC',clinical_class='lymphocyte',marker='single:CD3',fraction=None,pi_c=None,transfer_drift=None,bsccm_calibration=None,absolute_error=None,rq3_interpretation='raw only')])
        ev.csv_write(folder/'sanity.csv',[dict(teacher=f't{t}',seed=seed,arm=arm,slide=0,n=4,autofl_r=.6)])
        tr.write_json(folder/'eval.done.json',dict(done=True))
    ev.aggregate(synth)
    pm=ev.readcsv(synth/'paired_macros.csv')
    row=next(r for r in pm if r['comparison']=='H minus T' and r['summary']=='primary16')
    assert abs(float(row['mean'])-(.04-.005/3))<1e-8 and row['decision']=='H higher'
    assert abs(float(row['noise_floor'])-.02)<1e-8
    allrows=ev.readcsv(synth/'contrasts_all.csv')
    for r in allrows:
        if r['arm']=='H':r['auroc']=str(float(r['auroc'])-.03)
    test=lambda r:ev.contrasts.primary(r['domain'],r['contrast'])
    assert ev.seed_summary(allrows,'H','T',test)['decision']=='no difference'
    assert ev.seed_summary(allrows,'H_matched','T',test)['decision'].startswith('sensitivity only')
    macros=ev.readcsv(synth/'macro_means.csv');assert all(r['mean']=='' for r in macros if r['arm'].endswith('CCOT') and r['summary']=='all25')
    assert len(ev.contrasts.PRIMARY_PBC)==4 and len(ev.contrasts.PRIMARY_MLL)==12
    print('PASS 16 complete seeded runs, equal teacher weighting, primary16, seed floor claim/no-difference and matched sensitivity',flush=True)
    # Closed form matching and source-only controls, serialization, shared support.
    for target in [.02,.2,.8]:
        weight=tr.matched_weight(target,3.,7.)
        assert abs(weight*7/(3+weight*7)-target)<1e-12
    q=torch.randn(40,5);src=torch.randn(50,5)
    for kind in ['mean','CORAL','kNN']:
        mapper=ot.TrivialAlignment(kind).fit(q,src);got=mapper.transform(q[:3])
        assert torch.allclose(got,ot.TrivialAlignment.load(mapper.state()).transform(q[:3]))
        assert torch.isfinite(got).all() and torch.allclose(got.norm(dim=1),torch.ones(3),atol=1e-6)
    for tau in [.91,.3]:
        mapper=ot.UOT(rho=.05*tau/(1-tau)).fit(q,src)
        assert abs(mapper.diagnostics['tau']-tau)<1e-10
    aligned=ev.score_fractions(pp,np.array(['lymphocyte']*120),names,dict(primary=missing,proxy=ts),aligned=True)
    assert all(r['rq3_interpretation']=='not interpretable (shrinkage toward source)' for r in aligned)
    r=next(r for r in aligned if r['analysis']=='proxy' and r['marker']=='single:CD3')
    assert r['transfer_drift']==abs(r['fraction']-r['pi_c']) and r['bsccm_calibration']==abs(r['pi_c']-.7)
    print('PASS gradient budget algebra, all alignment controls and both fixed taus, separate RQ3 errors',flush=True)
    # Single cache file, committed-batch SIGTERM interruption and no duplicate inference.
    class FakePredictor:
        def __init__(self):self.ckpt=out/'fake.pt';self.ckpt.write_bytes(b'fixture');self.names=['m'];self.calls=0
        def predict(self,batch):
            self.calls+=1
            if self.calls==2:raise SystemExit(143)
            v=np.asarray(batch,np.float32)[:,None]
            return v,np.tile(v,(1,2)),np.tile(v[:,None],(1,8,2))
    fakepred=FakePredictor();cp=out/'single.cache.sqlite'
    try:ev.cache_predict(cp,fakepred,np.arange(5),lambda i:i,2,'source_train')
    except SystemExit as e:assert e.code==143
    fakepred.calls=2
    cached=ev.cache_predict(cp,fakepred,np.arange(5),lambda i:i,2,'source_train')
    assert fakepred.calls==4 and np.array_equal(cached['pred'][:,0],np.arange(5))
    assert not list(out.glob('*.parts')) and list(out.glob('single.cache*'))==[cp]
    print('PASS transactional single-file inference interruption/resume, no retained batch files',flush=True)
    # Actual source images + permitted PBC A anchors through training. Two
    # executions compare uninterrupted and SIGTERM/resumed parameter equality.
    common=[sys.executable,'-I','-B',str(HERE/'n056_hub_student.py'),'--device','cpu','--tiny','--steps','4','--hub-steps','4','--bs','2','--hub-bs','8','--threads','2']
    fresh=out/'fresh';resume=out/'resumed'
    if fresh.exists() or resume.exists():raise ValueError('smoke output must be fresh for signal check')
    subprocess.run(common+['--out',str(fresh)],check=True)
    proc=subprocess.Popen(common+['--out',str(resume)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        print(line.rstrip(),flush=True)
        if line.startswith('student update 1'):
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
    # Match-arm replay exercises original ridge/SigLIP/output/CE terms on real source.
    matched=out/'matched'
    subprocess.run(common+['--arm','H_matched','--out',str(matched)],check=True)
    logs=json.loads((matched/'gradient_norms.json').read_text())
    assert logs['calibration_steps']==4
    for r in logs['records']:
        assert set(r['reference'])=={'sig','dist','anchor'} and abs(r['T_share']-r['H_share'])<1e-12
    print('PASS real-source H_matched T-recipe replay and measured equal gradient shares',flush=True)
    matched_resume=out/'matched_resumed'
    proc=subprocess.Popen(common+['--arm','H_matched','--out',str(matched_resume)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        print(line.rstrip(),flush=True)
        if line.startswith('student update 1'):
            proc.send_signal(signal.SIGTERM);break
    print(proc.communicate(timeout=120)[0],end='',flush=True);assert proc.returncode==143
    subprocess.run(common+['--arm','H_matched','--out',str(matched_resume)],check=True)
    fresh_ck=torch.load(matched/'hub_student.pt',weights_only=False,map_location='cpu')
    resume_ck=torch.load(matched_resume/'hub_student.pt',weights_only=False,map_location='cpu')
    assert all(torch.equal(fresh_ck['student'][k],resume_ck['student'][k]) for k in fresh_ck['student'])
    assert json.loads((matched_resume/'gradient_norms.json').read_text())==logs
    print('PASS H_matched SIGTERM resume includes reference optimizer/RNG/calibration and identical shares',flush=True)
    other_seed=out/'seed1'
    subprocess.run(common+['--seed','1','--out',str(other_seed)],check=True)
    seed1=torch.load(other_seed/'hub_student.pt',weights_only=False,map_location='cpu')
    assert all(torch.equal(x['hub'][k],seed1['hub'][k]) for k in x['hub'])
    assert any(not torch.equal(x['student'][k],seed1['student'][k]) for k in x['student'])
    print('PASS two H student seeds differ with identical teacher-paired hub',flush=True)

    # Real full T checkpoint loading and CPU inference on TWO BSCCM train cells.
    from types import SimpleNamespace
    opts=SimpleNamespace(device='cpu',arm='T',data=str(tr.ROOT),seed=0)
    predictor=ev.Predictor(opts,pairs[0]);t=np.load(tr.ROOT/'targets_v4c.npz');rr=np.flatnonzero((t['hist_row']>=0)&(t['split']==0))[:2]
    hist=np.load(tr.ROOT/'pack/histology.npy',mmap_mode='r')
    batch=[np.clip(np.asarray(hist[t['hist_row'][i]],np.float32)/predictor.scale,0,1) for i in rr]
    p,z,zv=predictor.predict(batch)
    assert p.shape==(2,13) and z.shape==(2,256) and zv.shape==(2,8,256) and np.isfinite(p).all()
    mean=zv.mean(1);mean/=np.linalg.norm(mean,axis=1,keepdims=True)
    assert np.allclose(mean,z,atol=1e-6)
    class MeanOnlyMap:
        def transform(self,x):
            assert x.ndim==2 and x.shape[1]==256
            return x
    assert np.allclose(predictor.aligned(z,MeanOnlyMap()),predictor.decode(torch.tensor(z)).numpy(),atol=1e-6)
    print('PASS actual t0 T checkpoint, 8-view inference on 2 source training cells; no clinical test predictions',flush=True)
    print('ALL CPU SMOKE CHECKS PASSED',flush=True)
if __name__=='__main__':main()
