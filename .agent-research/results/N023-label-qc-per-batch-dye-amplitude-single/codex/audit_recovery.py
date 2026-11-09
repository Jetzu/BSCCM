"""Exploratory assay recovery control; no saved labels/models changed."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import nnls
W=Path('/hnvme/workspace/v181bb10-bsccm'); t=np.load(W/'targets_v4.npz'); sl,b,sp,c3=[t[k] for k in ('slide','batch','split','cls3')]
sm=pd.read_parquet(W/'bsccm_sm.parquet'); ix=pd.read_parquet(W/'bsccm_index.parquet'); ab=ix.antibodies.values
raw=sm[[c for c in sm if c.endswith('_shading_corrected')]].values
# Independent gate classifier uses exactly the 3 non-A fluorescence channels.
x0=sm[[c+'_full_model_unmixed' for c in ('CD3/CD19/CD56','CD45','CD16')]].values; x=np.zeros_like(x0); ar=np.flatnonzero(ab=='all')
for s in np.unique(sl[ar]):
    r=ar[sl[ar]==s]; ref=r[sp[r]==0]
    for j in range(3):
        q=np.quantile(np.log1p(x0[ref,j]),[.25,.5,.75]); x[r,j]=(np.log1p(x0[r,j])-q[1])/max(q[2]-q[0],1e-6)
tr=(b==0)&(c3>=0)&(sp==0)
clf=RandomForestClassifier(n_estimators=80,max_depth=10,min_samples_leaf=15,random_state=0,n_jobs=1).fit(x[tr],c3[tr])
rng=np.random.default_rng(23)
for bb in (0,1):
    spectra={}
    for m in ['CD123','HLA-DR','CD14','CD3','CD19','CD56','CD45','CD16','autofluor']:
        a='unstained' if m=='autofluor' else m; sel=(ab==a)&(b==bb)
        col=f'selection_example_{m}_positive_cells_batch_{bb}'
        if col not in sm: continue
        pos=sm[col].fillna(False).values.astype(bool)&sel
        d=np.mean(raw[pos],axis=0)-np.mean(raw[sel&~pos],axis=0); d=np.maximum(d,0); spectra[m]=d/np.linalg.norm(d)
    # No example autofl batch1 gate supplied; use unstained median as an explicit alternative.
    if 'autofluor' not in spectra:
        z=np.maximum(np.median(raw[(ab=='unstained')&(b==bb)],axis=0),0); spectra['autofluor']=z/np.linalg.norm(z)
    for s in np.unique(sl[ar[b[ar]==bb]]):
        r=np.flatnonzero((sl==s)&(sp==2)); r=rng.permutation(r)[:1500]; p=clf.predict_proba(x[r]); cls=np.where(p.max(1)>.8,p.argmax(1),-1)
        for version,yy in [('raw_A',sm['CD123/HLA-DR/CD14_full_model_unmixed'].values),('v3',t['Y_v3'][:,8]),('v4',t['Y'][:,8])]:
            y1,y0=yy[r[cls==2]],yy[r[cls==0]]; dd=(y1.mean()-y0.mean())/np.sqrt((y1.var()+y0.var())/2)
            print('ORIGINAL CONTRAST',bb,str(t['slide_names'][s]),version,'d',dd)
        for adir in ['CD14','HLA-DR','CD123','mean']:
            a=spectra[adir] if adir!='mean' else np.mean([spectra[m] for m in ['CD14','HLA-DR','CD123']],axis=0)
            ll=np.mean([spectra[m] for m in ['CD3','CD19','CD56']],axis=0)
            for af in (False,True):
                cols=[a,ll,spectra['CD45'],spectra['CD16']]+([spectra['autofluor']] if af else [])
                S=np.stack(cols,axis=1); coef=np.array([nnls(S,z)[0][0] for z in raw[r]])
                y1,y0=coef[cls==2],coef[cls==0]; d=(y1.mean()-y0.mean())/np.sqrt((y1.var()+y0.var())/2)
                print('REFIT',bb,str(t['slide_names'][s]),'A_direction',adir,'AF',af,'condition',np.linalg.cond(S),'d_mono_lym',d,'n_mono_lym',len(y1),len(y0))
    aa=sm['CD123/HLA-DR/CD14_full_model_unmixed'].values
    for a in ['all','unstained','CD14','HLA-DR','CD123','CD16','CD45']:
        r=(b==bb)&(ab==a); z=aa[r]
        print('A NEGATIVE/POSITIVE CONTROL',bb,a,'q50_q90_q99',np.quantile(z,[.5,.9,.99]).tolist())
