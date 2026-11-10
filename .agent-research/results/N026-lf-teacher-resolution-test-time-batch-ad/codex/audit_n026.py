"""Read-only N026 audit; outputs only beside this script."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import ast,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
torch.set_num_threads(1); torch.set_num_interop_threads(1)
W=Path('/hnvme/workspace/v181bb10-bsccm'); OUT=Path(__file__).resolve().parent
T=dict(np.load(W/'targets_v5.npz',allow_pickle=True)); Y,M=T['Y'],T['mask']; names=list(T['names'])
common=[j for j,n in enumerate(names[:12]) if n!='grp:CD123/HLA-DR/CD14']
print('COMMON',[str(names[j]) for j in common],flush=True)
def score(P,G,train_formula=False):
    rs=[]
    for j in common:
        v=M[G,j]; x=P[v,j]; y=Y[G[v],j]
        assert v.sum()>20 and np.isfinite(x).all() and np.isfinite(y).all()
        if train_formula:
            x=x-x.mean(); y=y-y.mean(); rs.append(float((x*y).sum()/(np.sqrt((x*x).sum()*(y*y).sum())+1e-9)))
        else: rs.append(float(np.corrcoef(x,y)[0,1]))
    return np.array(rs)
pred={}; configs={}; hashes={}; measures=[]; details={}
for family in ('n023_v5','n026_zoom2','n026_dual2','n026_nomp','n026_zoom2nomp'):
    for seed in ([0,1] if family in ('n023_v5','n026_zoom2') else [0]):
        for d in ('b10','b01'):
            name=f'{family}_{d}'+('_s1' if seed else ''); p=W/'runs'/name
            R=json.load(open(p/'result.json')); a=R['args']; configs[name]=a
            E=np.load(p/'eval_preds.npz'); P,G=E['P'],E['G']; pred[name]=(P,G)
            S=torch.load(p/'ckpt.pt',map_location='cpu',weights_only=False)
            has=(M.any(1)|(T['cls3']>=0))&(T['hist_row']<0); tb=0 if d=='b10' else 1
            tr=np.flatnonzero((T['batch']==1-tb)&has); ev=np.flatnonzero((T['batch']==tb)&(T['split']==2)&has)
            assert np.array_equal(G,ev) and np.array_equal(S['tr'],tr) and np.array_equal(S['ev'],ev)
            assert len(np.unique(G))==len(G) and not np.intersect1d(G,tr).size and P.shape==(len(G),len(names))
            defaults={'stem':'std','zoom':1,'dual_zoom':1}
            cfg={k:v for k,v in a.items() if k not in ('workers','out','adabn') and not (k in defaults and v==defaults[k])}
            assert cfg==S['cfg']
            h=hashlib.sha256()
            for k,v in S['m'].items(): h.update(k.encode()); h.update(v.numpy().tobytes())
            hashes[name]=h.hexdigest(); rs=score(P,G); r=float(rs.mean()); rr=score(P,G,True).mean()
            valid=M[tr].sum(0)>=1000; valid_indices=[j for j in range(12) if valid[j] and M[G,j].sum()>20]
            assert valid_indices==common
            delta=rr-R['final']['r_mean_markers_valid']; assert abs(delta)<1e-6
            steps=10*(len(tr)//a['bs']); bn=int(S['m']['enc.bn1.num_batches_tracked'])
            cfgdiff={k:v for k,v in cfg.items() if k not in ('protocol','seed') and configs.get('n023_v5_b10',{}).get(k,defaults.get(k))!=v}
            print('RUN',name,'seed',a['seed'],'ep',S['ep'],'tr/ev',len(tr),len(ev),'r',round(r,9),'train_score_delta',round(float(delta),10),'bn_count',bn,'steps',steps,'cfgdiff',cfgdiff,'hash',hashes[name][:12],flush=True)
            print('HISTORY',name,[(x['epoch'],round(x['train_loss'],4),round(x['r_mean_markers_valid'],4)) for x in R['log']],flush=True)
            measures.append(dict(arm=family,seed=seed,direction=d,value=r))
            details[name]={'cfg':cfg,'n_train':len(tr),'n_eval':len(ev),'BN_count':bn,'steps':steps,'r':r,'per_target':dict(zip(map(str,[names[j] for j in common]),map(float,rs))),'model_sha256':hashes[name]}
            ad=p/'eval_preds_adabn.npz'
            if ad.exists():
                A=np.load(ad); assert np.array_equal(A['G'],G)
                ar=score(A['P'],G).mean(); tgt=np.random.default_rng(0).permutation(np.flatnonzero((T['batch']==tb)&(T['split']==0)&has))[:20000]
                assert not np.intersect1d(tgt,G).size
                print('ADABN',name,'r',round(float(ar),9),'delta',round(float(ar-r),9),'rows',len(tgt),'ev_overlap',np.intersect1d(tgt,G).size,flush=True)
            del S
assert len(set(hashes.values()))==len(hashes)
print('ALL checkpoint state hashes distinct; checkpoint rows/config and saved metrics verified')
arms={}
for seed in (0,1):
    suff='_s1' if seed else ''
    arms[f'multiscale_s{seed}']=[(f'n023_v5_{d}{suff}',f'n026_zoom2_{d}{suff}') for d in ('b10','b01')]
    arms[f'cross_scale_s{seed}']=[(f'n023_v5_{d}{suff}',f'n026_zoom2_{d}'+('' if seed else '_s1')) for d in ('b10','b01')]
arms['same_base']=[(f'n023_v5_{d}',f'n023_v5_{d}_s1') for d in ('b10','b01')]
arms['same_zoom']=[(f'n026_zoom2_{d}',f'n026_zoom2_{d}_s1') for d in ('b10','b01')]
arm_scores={}
for arm,combos in arms.items():
    vs=[]
    for combo in combos:
        ps=[pred[n] for n in combo]; G=ps[0][1]; assert all(np.array_equal(g,G) for _,g in ps)
        r=float(score(np.mean([p for p,_ in ps],axis=0),G).mean()); vs.append(r)
        measures.append(dict(arm=arm,seed=-1,direction='b10' if 'b10' in combo[0] else 'b01',value=r))
    arm_scores[arm]=float(np.mean(vs)); print('ENSEMBLE',arm,'directions',vs,'GOAL',arm_scores[arm],flush=True)
base=[]; zoom=[]
for seed in (0,1):
    suff='_s1' if seed else ''
    base.append(np.mean([details[f'n023_v5_{d}{suff}']['r'] for d in ('b10','b01')]))
    zoom.append(np.mean([details[f'n026_zoom2_{d}{suff}']['r'] for d in ('b10','b01')]))
print('NOISE single_run sample_sd base/zoom',float(np.std(base,ddof=1)),float(np.std(zoom,ddof=1)),'pooled',float(np.sqrt((np.var(base,ddof=1)+np.var(zoom,ddof=1))/2)))
print('NOISE paired scale deltas',list(np.array(zoom)-base),'paired multiscale gain_vs_own_base',[arm_scores[f'multiscale_s{i}']-base[i] for i in range(2)])
source=Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py').read_text(); tree=ast.parse(source)
nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name=='_cfg') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NEW_DEFAULTS' for t in n.targets))]
ns={'a':SimpleNamespace(**dict({'stem':'std','zoom':1,'dual_zoom':1,'adabn':0},**configs['n023_v5_b10']))}; exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual_cfg>','exec'),ns); default=ns['_cfg']()
for key,value in [('zoom',2),('stem','nomaxpool'),('dual_zoom',2),('adabn',20000),('seed',9),('targets','targets_v6.npz')]:
    old=getattr(ns['a'],key); setattr(ns['a'],key,value); print('RESUME change',key,value,'accepted',ns['_cfg']()==default); setattr(ns['a'],key,old)
print('TARGET metadata keys',sorted(T.keys()))
json.dump(dict(runs=details,ensembles=arm_scores,configs=configs),open(OUT/'audit_metrics.json','w'),indent=2)
with open(OUT/'audit_metrics.csv','w') as f:
    wr=csv.DictWriter(f,fieldnames=['arm','seed','direction','value']); wr.writeheader(); wr.writerows(measures)
