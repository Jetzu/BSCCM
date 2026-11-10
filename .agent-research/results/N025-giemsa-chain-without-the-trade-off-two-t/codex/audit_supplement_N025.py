from pathlib import Path
import json, numpy as np, pandas as pd
R=Path('/home/hpc/v181bb/v181bb10/BSCCM');W=Path('/hnvme/workspace/v181bb10-bsccm')
O=R/'.agent-research/results/N025-giemsa-chain-without-the-trade-off-two-t'
T={v:dict(np.load(W/f'targets_v{v}.npz',allow_pickle=True)) for v in (4,5,6)}
B=T[4];h=B['hist_row'];sp=B['split'];b=B['batch']
print('=== CROSS SLIDE RECOMPUTE / CONFIG ===')
for row in pd.read_csv(O/'n025_metrics.csv').itertuples():
 if not row.metric.startswith('gs_autofl_xslide'):continue
 d=json.loads((W/'runs'/row.run/'result.json').read_text());args=d['args']
 if not (W/'runs'/row.run/'test_preds.npz').exists():
  print(row.run,'MISSING test_preds; stored r',row.value,'args',args);continue
 E=np.load(W/'runs'/row.run/'test_preds.npz');g=E['gidx']
 bb=0 if args['holdout']=='batch0' else 1
 expect=np.flatnonzero((h>=0)&(b==bb));expect=expect[np.argsort(h[expect])]
 assert np.array_equal(g,expect)
 y=T[int(args.get('targets','targets_v4.npz')[-5])]['Y'][g,12]
 r=np.corrcoef(E['student'][:,12],y)[0,1]
 print(row.run,'n',len(g),'splits',np.unique(sp[g],return_counts=True),'r',r,'stored',row.value,'args',args)
print('=== V6 CHANGES / RIDGE SAMPLING ===')
for v in (4,5,6):
 tt=T[v];j=12;r=np.flatnonzero((sp==0)&tt['mask'][:,j]&(h<0));r=np.random.default_rng(j).permutation(r)[:60000]
 print('ridge fit',v,'n',len(r),'GS',int((h[r]>=0).sum()),'by batch',np.unique(b[r],return_counts=True),'by slide',np.unique(tt['slide'][r],return_counts=True))
print('=== FOV BOUND METRICS BATCH PICK ===')
E=np.load(W/'runs/n025_tt6_s0/test_preds.npz');g=E['gidx'];tb=int(b[g[0]])
print('bound tb',tb,'GS batch counts',np.unique(b[g],return_counts=True))
print('=== CIA CONSERVATIVE COUNTEREXAMPLE ===')
# Standardised orthogonal vectors; exact correlations, no Monte Carlo uncertainty.
q=np.linalg.qr(np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]]))[0];X=q[:,0];E=q[:,1]
teacher=X;student=.8*X+.6*E;marker=.8*X-.6*E
rST=np.corrcoef(student,teacher)[0,1];rTM=np.corrcoef(teacher,marker)[0,1];rSM=np.corrcoef(student,marker)[0,1]
print('rST',rST,'rTM',rTM,'product',rST*rTM,'actual rSM',rSM)
