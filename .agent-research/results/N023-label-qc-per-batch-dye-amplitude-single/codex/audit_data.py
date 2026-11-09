"""Read-only N023 audit. Run from codex; outputs only beside this file."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
W = Path('/hnvme/workspace/v181bb10-bsccm')
O = Path(__file__).resolve().parent
A = dict(np.load(W/'targets_v4.npz', allow_pickle=True))
B = dict(np.load(W/'targets_v5.npz', allow_pickle=True))
Y, V, M, N = A['Y'], B['Y'], A['mask'], B['mask']
b, sl, sp, hr, c3 = [A[k] for k in ('batch','slide','split','hist_row','cls3')]
names, sn = A['names'].astype(str), A['slide_names'].astype(str)
ix, sm = pd.read_parquet(W/'bsccm_index.parquet'), pd.read_parquet(W/'bsccm_sm.parquet')
print('ROW ALIGNMENT', np.array_equal(ix.global_index, np.arange(len(Y))), np.array_equal(sm.global_index, np.arange(len(Y))))
print('SHAPE', Y.shape, 'names', names.tolist())
print('UNCHANGED KEYS', {k:np.array_equal(A[k],B[k],equal_nan=True) if A[k].dtype.kind not in 'OUS' else np.array_equal(A[k],B[k]) for k in A if k not in ('Y','mask')})
print('Y_v4 exact', np.array_equal(B['Y_v4'],Y), 'finite',np.isfinite(V).all(), 'masked zeros', np.all(V[~N]==0))
print('mask additions',int((N&~M).sum()),'removals',int((M&~N).sum()),'qc exact',np.array_equal(B['qc_masked'],M&~N))
rows=[]
for j,n in enumerate(names):
    for s in np.unique(sl):
        r = (sl==s)&M[:,j]
        if not r.any(): continue
        common = r&N[:,j]; x,y=Y[common,j],V[common,j]
        rows.append(dict(target=n,slide=sn[s],n_v4=int(r.sum()),n_v5=int((r&N[:,j]).sum()),changed=int((r&(Y[:,j]!=V[:,j])).sum()),n_cal=int((r&(sp==0)).sum()),n_cal_hist=int((r&(sp==0)&(hr>=0)).sum()),corr=float(np.corrcoef(x,y)[0,1]) if len(x)>2 else np.nan,mean_v4=float(Y[r,j].mean()),mean_v5=float(y.mean()) if len(y) else np.nan,std_v4=float(Y[r,j].std()),std_v5=float(y.std()) if len(y) else np.nan))
df=pd.DataFrame(rows); df.to_csv(O/'target_diff.csv',index=False)
print('DIFF BY TARGET\n'+df.groupby('target')[['n_v4','n_v5','changed']].sum().to_string())
print('autofl per slide\n'+df[df.target=='autofl'].to_string(index=False))
R=Y.astype(float).copy(); RM=M.copy(); expected=(ix.antibodies.values=='all')&(b==1)
jA=list(names).index('grp:CD123/HLA-DR/CD14'); RM[expected,jA]=False
for j,n in enumerate(names):
    if n.startswith('grp:') or n=='autofl':
        for bb in (0,1):
            rr=np.flatnonzero(RM[:,j]&(b==bb)); ss=np.unique(sl[rr])
            if len(ss)<2: continue
            for s in ss:
                r=rr[sl[rr]==s]; ref=r[sp[r]==0]
                q=np.quantile(R[ref,j],[.25,.5,.75]); scale=max((q[2]-q[0])/1.349,1e-6)
                R[r,j]=(R[r,j]-q[1])/scale
            ref=rr[sp[rr]==0]; R[rr,j]=(R[rr,j]-np.mean(R[ref,j]))/np.std(R[ref,j])
R[~RM]=0
print('RECONSTRUCTION mask exact',np.array_equal(RM,N),'float32 exact',np.array_equal(R.astype('float32'),V),'maxabs',np.max(np.abs(R.astype('float32')-V)))
metrics=[]
for run in ['n014_sslinit_b1to0','n014_sslinit_b0to1','n022_base_b10']:
    e=np.load(W/'runs'/run/'eval_preds.npz'); p,g=e['P'],e['G']; cfg=json.load(open(W/'runs'/run/'result.json'))['args']
    tb=1 if cfg['protocol']=='b1to0' else 0
    has=N.any(1)|(c3>=0)
    if not cfg['include_gs']: has &= hr<0
    tr=np.flatnonzero((b==tb)&has)
    if cfg['max_train']: tr=np.random.default_rng(0).permutation(tr)[:cfg['max_train']]
    trained=N[tr].sum(0)>=1000
    print('RUN',run,'cfg_protocol',cfg['protocol'],'eval batches',np.unique(b[g]).tolist(),'eval splits',np.unique(sp[g]).tolist(),'counting rule matches',np.array_equal(trained,N[b==tb].sum(0)>=1000))
    vals={}
    for version,yy,mm in [('v4',Y,M),('v5',V,N)]:
        rs={}
        for j,n in enumerate(names[:12]):
            r=mm[g,j]
            if r.sum()>20: rs[n]=float(np.corrcoef(p[r,j],yy[g[r],j])[0,1])
        common=[v for n,v in rs.items() if n!=names[jA]]
        valid=[rs[n] for j,n in enumerate(names[:12]) if n in rs and trained[j]]
        vals[version]=dict(all=float(np.mean(list(rs.values()))),common11=float(np.mean(common)),valid=float(np.mean(valid)),n_available=len(rs),n_valid=len(valid),r=rs)
    print('RESCORE',run,json.dumps(vals)); metrics.append(dict(run=run,**vals))
json.dump(metrics,open(O/'audit_rescore.json','w'),indent=2)
allrows=np.flatnonzero(ix.antibodies.values=='all'); rawcols=[n+'_full_model_unmixed' for n in ('CD3/CD19/CD56','CD45','CD16')]
X0=sm[rawcols].values
for method in ('robust_log','ranks'):
    X=np.zeros((len(Y),3))
    for s in np.unique(sl[allrows]):
        r=allrows[sl[allrows]==s]; ref=r[sp[r]==0]
        for j in range(3):
            if method=='robust_log':
                z=np.log1p(X0[r,j]); zref=np.log1p(X0[ref,j]); q=np.quantile(zref,[.25,.5,.75]); X[r,j]=(z-q[1])/max(q[2]-q[0],1e-6)
            else:
                refsort=np.sort(X0[ref,j]); X[r,j]=np.searchsorted(refsort,X0[r,j],side='right')/len(refsort)
    tr=(b==0)&(c3>=0)&(sp==0); ev=(b==0)&(c3>=0)&(sp==2)
    clf=RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=15,random_state=0,n_jobs=1).fit(X[tr],c3[tr])
    print('INDEPENDENT GATES',method,'b0 acc',clf.score(X[ev],c3[ev]),'balanced',balanced_accuracy_score(c3[ev],clf.predict(X[ev])))
    p=clf.predict_proba(X[allrows]); cls=np.where(p.max(1)>.8,p.argmax(1),-1)
    for s in np.unique(sl[allrows]):
        for split in ['all','test']:
            r=(sl[allrows]==s)&((sp[allrows]==2) if split=='test' else True); g=allrows[r]; k=cls[r]; means=[float(Y[g[k==c],jA].mean()) for c in range(3)]; z1,z0=Y[g[k==2],jA],Y[g[k==0],jA]
            d=(z1.mean()-z0.mean())/np.sqrt((z1.var()+z0.var())/2)
            print('GATE CONTRAST',method,sn[s],split,'counts',[int((k==c).sum()) for c in range(3)],'means',means,'d',d)
raw=[c for c in sm if c.endswith('_shading_corrected')]; rawarr=sm[raw].values; ab=ix.antibodies.values
sig={}
for marker in ['CD123','HLA-DR','CD14','CD3','CD19','CD56','CD45','CD16']:
    for bb in (0,1):
        sel=(ab==marker)&(b==bb); pos=sm[f'selection_example_{marker}_positive_cells_batch_{bb}'].fillna(False).values.astype(bool)&sel
        neg=sel&~pos
        v=np.mean(rawarr[pos],axis=0)-np.mean(rawarr[neg],axis=0); vv=np.maximum(v,0); sig[bb,marker]=vv
        u=sm[f'{marker}_single_antibody_model_unmixed'].values[sel]; lo=u<np.nanpercentile(u,40)
        print('DYE CHECK',marker,bb,'n_positive',int(pos.sum()),'strict bottom40 n',int(lo.sum()),'raw_gate_delta',v.tolist(),'clipped_norm',np.linalg.norm(vv))
    print('DYE GATE RATIO',marker,np.linalg.norm(sig[1,marker])/np.linalg.norm(sig[0,marker]))
print('GS CALIBRATION OVERLAP')
for bb in (0,1):
    r=(b==bb)&(hr>=0)&N[:,-1]
    print('batch',bb,'histology eval all',int(r.sum()),'split0 in normalization',int((r&(sp==0)).sum()),'slide ids',np.unique(sl[r]).tolist())
