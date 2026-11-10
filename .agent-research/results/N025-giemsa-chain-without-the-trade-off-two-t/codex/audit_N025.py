"""Read-only N025 audit: only stdout; all arithmetic single-threaded."""
import ast, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score
R=Path('/home/hpc/v181bb/v181bb10/BSCCM')
W=Path('/hnvme/workspace/v181bb10-bsccm')
O=R/'.agent-research/results/N025-giemsa-chain-without-the-trade-off-two-t'
T={v:dict(np.load(W/f'targets_v{v}.npz',allow_pickle=True)) for v in (4,5,6)}
B=T[4]; ja=list(B['names']).index('autofl')
hist,sp,slide,batch=(B[k] for k in ('hist_row','split','slide','batch'))
arms={'v4r18':['n021_gs_anc','n021_gs_anc_s1','n024_ref_gs_anc_s2'],
'v5r18':[f'n024_gs_anc_s{s}' for s in range(3)], 'tt5':[f'n025_tt5_s{s}' for s in range(3)],
'v6r18':[f'n025_v6_s{s}' for s in range(3)],'tt6':[f'n025_tt6_s{s}' for s in range(3)]}
def read(run):return json.loads((W/'runs'/run/'result.json').read_text())
def corr(x,y):return float(np.corrcoef(np.asarray(x,float),np.asarray(y,float))[0,1])
def ci(x):
 x=np.asarray(x,float);h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
 return [float(x.mean()-h),float(x.mean()+h)]
print('=== LABEL ALIGNMENT, MASKS, CALIBRATION ===')
for v in (5,6):print('alignment',v,{k:np.array_equal(T[v][k],B[k]) for k in ('names','split','hist_row','slide','slide_names','batch','fov')})
print('v5/v6 markers exactly equal',np.array_equal(T[5]['Y'][:,:12],T[6]['Y'][:,:12]),np.array_equal(T[5]['mask'][:,:12],T[6]['mask'][:,:12]))
for v,tt in T.items():
 m=tt['mask'][:,ja]
 print('autofl counts',v,'total',int(m.sum()),'nonGS',int((m&(hist<0)).sum()),'ridge split0 nonGS',int((m&(hist<0)&(sp==0)).sum()),'GS',int((m&(hist>=0)).sum()))
 has=(tt['mask'].any(1)|(tt['cls3']>=0))&(hist<0)
 print('LF eligible',v,'fov train',int((has&(sp==0)).sum()),'fov eval',int((has&(sp==2)).sum()))
for bb in (0,1):
 ref=T[6]['mask'][:,ja]&(batch==bb)&(sp==0);gs=ref&(hist>=0);ng=ref&(hist<0);y=T[5]['Y'][:,ja]
 print('v6 norm',bb,'all ref',int(ref.sum()),'GS ref',int(gs.sum()),'nonGS ref',int(ng.sum()),'all mean/sd',y[ref].mean(),y[ref].std(),'nonGS mean/sd',y[ng].mean(),y[ng].std())
for s in np.unique(slide[hist>=0]):
 sel=(slide==s)&B['mask'][:,ja]
 print('GS slide',int(s),str(B['slide_names'][s]),'GS by split',[int(((hist>=0)&sel&(sp==p)).sum()) for p in (0,1,2)])
 for v in (5,6):
  x,y=B['Y'][sel,ja].astype(float),T[v]['Y'][sel,ja].astype(float)
  coef=np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y,rcond=None)[0]
  print('affine v4->v'+str(v),'slope/intercept',coef.tolist(),'max_residual',float(np.max(np.abs(y-coef[0]*x-coef[1]))),'r',corr(x,y))
print('=== GS ROWS, METRIC REPRODUCTION, FAIRNESS ===')
gold=pd.read_csv(O/'n025_metrics.csv');refargs=read(arms['v4r18'][0])['args']
ignore={'out','seed','lf_feats','targets','aux_feats','aux_targets','aux_names'}
refg=None;results=[];maxerr=0.
for arm,runs in arms.items():
 for seed,run in enumerate(runs):
  res=read(run);args=res['args'];e=np.load(W/'runs'/run/'test_preds.npz');g=e['gidx']
  if refg is None:refg=g.copy()
  expected=np.flatnonzero((hist>=0)&(sp==2));expected=expected[np.argsort(hist[expected])]
  assert np.array_equal(g,expected) and np.array_equal(g,refg) and np.array_equal(hist[g],e['rows'])
  diff={k:[refargs.get(k),args.get(k)] for k in refargs.keys()|args.keys() if k not in ignore and refargs.get(k)!=args.get(k)}
  print('GS args',arm,seed,'non-treatment differences',diff,'seed',args['seed'],'final',res['final']['step'],'checkpoints',[q['step'] for q in res['log']])
  for s in np.unique(slide[g]):
   mm=slide[g]==s; vals=[corr(e['student'][mm,ja],T[v]['Y'][g[mm],ja]) for v in (4,5,6)]
   metric='gs_autofl_within_b'+str(batch[g[mm][0]])
   saved=gold[(gold.arm==arm)&(gold.seed==seed)&(gold.metric==metric)].value.iloc[0]
   maxerr=max(maxerr,abs(saved-vals[0]));results.append([arm,seed,metric,vals[0]])
   print('GS recompute',arm,seed,'slide',int(s),'n',int(mm.sum()),'r v4/v5/v6',vals)
  if seed==0:print('POOLED fixed prediction',arm,[corr(e['student'][:,ja],T[v]['Y'][g,ja]) for v in (4,5,6)])
print('max error vs n025_metrics GS',maxerr)
rr=pd.DataFrame(results,columns=['arm','seed','metric','r'])
print(rr.groupby(['metric','arm']).r.agg(['mean','std','count']).to_string())
for arm in ('v4r18','v5r18','v6r18'):
 args=read(arms[arm][0])['args'];path=Path(args['lf_feats']);teacher=json.loads((path.parent/'result.json').read_text());pp=json.loads((path/'probe.json').read_text());prows=np.load(path/'probe_rows.npz')
 print('LF teacher',arm,'args',teacher['args'],'n_train/eval',teacher['n_train'],teacher['n_eval'],'probe checkpoint',pp['args']['ckpt'],'probe GS rows',{k:int((hist[prows[k]]>=0).sum()) for k in prows.files})
for arm in ('tt5','tt6'):
 ref=np.load(W/'runs'/arms['v4r18'][0]/'test_preds.npz')['teacher'];e=np.load(W/'runs'/arms[arm][0]/'test_preds.npz')['teacher']
 print('teacher MK untouched',arm,'max abs diff',float(np.max(np.abs(ref[:,:12]-e[:,:12]))),'autofl variance',float(e[:,ja].var()),'v4 autofl variance',float(ref[:,ja].var()))
print('=== MLL23 INDEPENDENT AUROC FROM PREDICTIONS ===')
tree=ast.parse((R/'scripts/apply_gs_student.py').read_text())
node=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(q,ast.Name) and q.id=='MC' for q in n.targets))
mc=eval(compile(ast.Expression(node),'<literal contrast dictionary>','eval'),{'__builtins__':{}},{'L':['lymphocyte'],'N':['neutrophil_segmented']})
refs=('segneut>lym CD16','mono>lym+neut CD14','lym>neut CD3');contrasts={k:v for k,v in mc.items() if k not in refs};assert len(contrasts)==12
tr=pd.read_csv(O/'transfer_multiseed.csv');aucrows=[];maxerr=0.;clsref=None
for (arm,seed),sub in tr[tr['set']=='MLL23'].groupby(['arm','seed']):
 dirs=sub.dir.unique();assert len(dirs)==1
 d=O/dirs[0];e=np.load(d/'mll23_gs_student.npz');p,cls,names=e['pred'],e['cls'],list(e['names'])
 if clsref is None:clsref=cls.copy();print('MLL class counts',dict(zip(*[q.tolist() for q in np.unique(cls,return_counts=True)])))
 assert np.array_equal(cls,clsref) and len(sub)==15 and sub.contrast.nunique()==15
 for key,(marker,pos,neg,sign) in contrasts.items():
  mm=np.isin(cls,pos+neg);auc=roc_auc_score(np.isin(cls[mm],pos),sign*p[mm,names.index(marker)])
  saved=sub[sub.contrast==key].auroc.iloc[0];maxerr=max(maxerr,abs(auc-saved));aucrows.append([arm,seed,key,float(auc)])
 print('MLL predictions',arm,int(seed),'n',len(cls),'path',str(d))
aucs=pd.DataFrame(aucrows,columns=['arm','seed','contrast','auc']);means=aucs.groupby(['arm','seed']).auc.mean().unstack('arm')
print('max AUROC diff from rounded CSV',maxerr);print('unrounded MLL mean by seed\n'+means.to_string())
for arm in means:
 x=means[arm].to_numpy();print('MLL mean/spread',arm,'mean',x.mean(),'population sd',x.std(),'sample sd',x.std(ddof=1),'95% t CI',ci(x))
for arm in ('tt5','tt6'):
 d=(means[arm]-means.v4r18).to_numpy();print('paired difference',arm,'- v4',d.tolist(),'mean',d.mean(),'sample sd',d.std(ddof=1),'95% t CI',ci(d))
print('per contrast mean\n'+aucs.groupby(['contrast','arm']).auc.mean().unstack('arm').round(6).to_string())
print('PBC heldout contrasts\n'+tr[(tr['set']=='PBC-B')&tr.contrast.str.contains('held-out')].groupby(['contrast','arm']).auroc.agg(['mean','std']).to_string())
print('=== GOAL SAVED PREDICTIONS ===')
goals={5:json.loads((R/'.agent-research/results/N023-label-qc-per-batch-dye-amplitude-single/goal_runs_v5.json').read_text()),6:json.loads((O/'goal_runs_v6.json').read_text())}
for v,records in goals.items():
 vals=[];matched=[]
 for rec in records:
  run=rec['run'];res=read(run);e=np.load(W/'runs'/run/'eval_preds.npz');g=e['G'];tt=T[v];common=[k for k in range(12) if k!=8]
  rs=[corr(e['P'][tt['mask'][g,j],j],tt['Y'][g[tt['mask'][g,j]],j]) for j in common];value=float(np.mean(rs));vals.append(value)
  if (v==6 or run.startswith('n023_v5_')):matched.append(value)
  print('GOAL recompute',v,run,'r',value,'stored',rec['r_mean_common'],'eval rows',len(g),'GS eval',int((hist[g]>=0).sum()),'args',res['args'],'n_train',res['n_train'])
 print('GOAL mean TRAINED ON VERSION',v,float(np.mean(matched)),'matched run count',len(matched))
print('=== AUX INTERFACE PERMUTATION REPRODUCER ===')
names=list(B['names']);j=names.index('autofl');aux_names=B['names'][np.roll(np.arange(len(names)),1)]
print('rolled aux names; requested autofl; current code uses',str(aux_names[j]),'correct aux index',list(aux_names).index('autofl'),'primary index used',j)
