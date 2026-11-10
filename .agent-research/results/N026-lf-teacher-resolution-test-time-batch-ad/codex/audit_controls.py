"""Extra read-only controls: target contributions, pair noise, actual resume gate."""
import ast,json,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent; W=Path('/hnvme/workspace/v181bb10-bsccm'); T=np.load(W/'targets_v5.npz'); N=list(T['names']); J=[j for j in range(12) if 'CD123/HLA' not in N[j]]
R=json.load(open(D/'audit_metrics.json'))
base=np.array([np.mean([R['runs'][f'n023_v5_{d}'+('_s1' if s else '')]['r'] for d in ('b10','b01')]) for s in (0,1)])
z=np.array([np.mean([R['runs'][f'n026_zoom2_{d}'+('_s1' if s else '')]['r'] for d in ('b10','b01')]) for s in (0,1)])
same=R['ensembles']['same_zoom']
print('SINGLE goal means base',base.mean(),'zoom',z.mean())
print('MULTISCALE advantage over same_zoom',[R['ensembles'][f'multiscale_s{s}']-same for s in (0,1)])
delta=z-base; print('PAIRED zoom-base deltas',delta,'mean',delta.mean(),'sd',delta.std(ddof=1))
# df=1, t_0.975 = 12.706204736; this is descriptive only (n=2).
print('PAIRED zoom-base descriptive t95',delta.mean()+np.array([-1,1])*12.706204736*delta.std(ddof=1)/np.sqrt(2))
print('ENSEMBLE matched sample sd',np.std([R['ensembles'][f'multiscale_s{s}'] for s in (0,1)],ddof=1),'n=2; shared same_zoom comparator prevents independent gain replicates')
for d in ('b10','b01'):
    ps={}
    for f in ('n023_v5','n026_zoom2'):
        for s in (0,1):
            e=np.load(W/'runs'/(f'{f}_{d}'+('_s1' if s else ''))/'eval_preds.npz'); ps[f,s]=e['P']; G=e['G']
    baseline=(ps['n023_v5',0]+ps['n023_v5',1])/2; zoom=(ps['n026_zoom2',0]+ps['n026_zoom2',1])/2
    def rr(p,j):
        v=T['mask'][G,j]; return np.corrcoef(p[v,j],T['Y'][G[v],j])[0,1]
    print('TARGET_DELTA',d,'name,multi0-baseens,multi1-baseens,multi0-zoomens,multi1-zoomens')
    for j in J:
        multi=[(ps['n023_v5',s]+ps['n026_zoom2',s])/2 for s in (0,1)]
        print(str(N[j]),*[round(rr(p,j)-rr(ref,j),6) for ref in (baseline,zoom) for p in multi])
# Execute actual resume guard in isolation, changing label values but no config or rows.
p=Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py'); t=ast.parse(p.read_text())
block=next(n for n in ast.walk(t) if isinstance(n,ast.If) and n.lineno==132)
ns=dict(np=np,s={'cfg':{'targets':'targets_v5.npz'},'tr':np.array([1,2]),'ev':np.array([3])},cfg={'targets':'targets_v5.npz'},tr=np.array([1,2]),ev=np.array([3]),ck='synthetic.pt')
for labels in (np.array([0.,1.]),np.array([99.,-99.])):
    ns['Y']=labels; exec(compile(ast.Module(body=[block],type_ignores=[]),'<actual_resume_guard>','exec'),ns); print('RESUME unchanged path/rows with Y=',labels.tolist(),'ACCEPTED')
print('SOURCE SHA256')
for p in [p,Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/rescore_targets.py'),D.parent/'ensemble_scoring.py']:
    print(str(p),hashlib.sha256(p.read_bytes()).hexdigest())
