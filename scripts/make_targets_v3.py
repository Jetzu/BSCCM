"""Data curation v3 (N005 + red-team F4). Writes targets_v3.npz. ALL calibration statistics (q10, floor, unstained median/MAD,
z mean/sd) are fitted on TRAINING-split rows (split==0) of each batch only, then applied to every row: evaluation labels never
define the transform; for batch hold-out the held-out batch is calibrated on its own split-0 rows (disjoint from its split-2 eval rows).
Previous: targets_v2.npz (calibrated on all rows). All transforms documented here; raw columns kept in bsccm_sm.parquet.
Markers (masked; each cell only gets targets it actually measured):
  single:<M> (8)  cells with antibodies==M: x = M_single_antibody_model_unmixed; t = asinh(x / q10_{b,M}); z-scored within (batch, M).
  grp:<G>   (4)   'all' cells: x = G_full_model_unmixed; t = asinh((x - med_unst_{b,G}) / mad_unst_{b,G}) (calibrated to same-batch
                  unstained = autofluorescence + crosstalk floor); z-scored within (batch,'all').
  autofl     (1)  every non-'all' cell (incl. unstained / histology cells): autofluor_single_antibody_model_unmixed, asinh(x/q10_b), z within batch.
  censored mask: x at the per-(batch,column) floor (authors clip below the 0.5th pct of non-zero values).
Classes: cls10 = 10 gates (batch-0 'all' only; 13 double-gated -> -1); cls3 = lymphocyte {0}, granulocyte {1,2,3}, monocyte {7,8,9}; RBC(4)/unknown(5,6) -> -1.
Splits: split_fov (80/10/10 by FOV, seed 0, from v1); batch kept for batch-holdout protocols."""
import os as _os; CLEAN = _os.environ.get('BSCCM_CLEAN', '0') == '1'; S_ = 'c' if CLEAN else ''
CF = _os.environ.get('BSCCM_COFACTOR', 'q10'); TAG = '' if CF == 'q10' else f'_cf{CF}'   # N048: single-marker asinh cofactor (q10 = v3-v5 default; q50 | numeric fixed)   # N029: CLEAN=1 -> no Giemsa (hist_row>=0) cell enters ANY fit/calibration; outputs targets_v*c
import numpy as np, pandas as pd, json
W='/hnvme/workspace/v181bb10-bsccm'
idx=pd.read_parquet(f'{W}/bsccm_index.parquet'); sm=pd.read_parquet(f'{W}/bsccm_sm.parquet'); v1=np.load(f'{W}/targets.npz'); CAL=(v1['split']==0)&((v1['hist_row']<0) if CLEAN else True)
N=len(idx); ab=idx.antibodies.values; b=idx.batch.values
S8=['CD45','CD123','CD19','CD56','CD14','CD16','HLA-DR','CD3']; G4=['CD123/HLA-DR/CD14','CD3/CD19/CD56','CD45','CD16']
names=[f'single:{m}' for m in S8]+[f'grp:{g}' for g in G4]+['autofl']; K=len(names)
Y=np.zeros((N,K),np.float32); T=np.zeros((N,K),np.float32); Mk=np.zeros((N,K),bool); Cz=np.zeros((N,K),bool); info={}
def zfit(t,sel):
    c=sel&CAL; mu,sd=t[c].mean(),t[c].std()+1e-6; return (t-mu)/sd,float(mu),float(sd)
for j,m in enumerate(S8):
    x=sm[f'{m}_single_antibody_model_unmixed'].values
    for bb in (0,1):
        sel=(ab==m)&(b==bb)
        if not sel.any(): continue
        q10=np.percentile(x[sel&CAL],10); fl=x[sel&CAL].min()
        cof={'q10':q10,'q50':np.percentile(x[sel&CAL],50)}.get(CF) or float(CF); t=np.arcsinh(x/cof)
        z,mu,sd=zfit(t,sel); Y[sel,j]=z[sel]; T[sel,j]=t[sel]; Mk[sel,j]=True; Cz[sel,j]=x[sel]<=fl*(1+1e-6)
        info[f'{names[j]}|b{bb}']=dict(n=int(sel.sum()),q10=float(q10),floor=float(fl),censored=float(Cz[sel,j].mean()),mu=mu,sd=sd)
for k,g in enumerate(G4):
    j=8+k; x=sm[f'{g}_full_model_unmixed'].values
    for bb in (0,1):
        ref=(ab=='unstained')&(b==bb); sel=(ab=='all')&(b==bb)
        med=np.median(x[ref&CAL]); mad=np.median(np.abs(x[ref&CAL]-med))*1.4826+1e-6; fl=x[(b==bb)&CAL].min()
        t=np.arcsinh((x-med)/mad); z,mu,sd=zfit(t,sel); Y[sel,j]=z[sel]; T[sel,j]=t[sel]; Mk[sel,j]=True; Cz[sel,j]=x[sel]<=fl*(1+1e-6)
        info[f'{names[j]}|b{bb}']=dict(n=int(sel.sum()),unst_med=float(med),unst_mad=float(mad),censored=float(Cz[sel,j].mean()),
             frac_above_unst_p99=float((x[sel]>np.percentile(x[ref&CAL],99)).mean()),mu=mu,sd=sd)
x=sm['autofluor_single_antibody_model_unmixed'].values; j=K-1
for bb in (0,1):
    sel=(ab!='all')&(b==bb)&np.isfinite(x)
    q10=np.percentile(x[sel&CAL],10); fl=x[sel&CAL].min(); t=np.arcsinh(x/q10); z,mu,sd=zfit(t,sel)
    Y[sel,j]=z[sel]; T[sel,j]=t[sel]; Mk[sel,j]=True; Cz[sel,j]=x[sel]<=fl*(1+1e-6); info[f'autofl|b{bb}']=dict(n=int(sel.sum()),q10=float(q10),censored=float(Cz[sel,j].mean()))
gates=[c for c in sm.columns if c.startswith('selection_gated_')]; Gm=sm[gates].astype(bool).values; ng=Gm.sum(1)
cls10=np.where(ng==1,Gm.argmax(1),-1).astype(np.int16)
map3={0:0,1:1,2:1,3:1,7:2,8:2,9:2}; cls3=np.array([map3.get(int(c),-1) for c in cls10],np.int16)
np.savez(f'{W}/targets_v3{S_}{TAG}.npz',Y=Y,T=T,mask=Mk,censored=Cz,names=np.array(names),cls10=cls10,cls3=cls3,cls3_names=np.array(['lymphocyte','granulocyte','monocyte']),
         gates=np.array([g.replace('selection_gated_','') for g in gates]),split=v1['split'],fov=v1['fov'],hist_row=v1['hist_row'],batch=b,
         slide=pd.factorize(idx.data_path)[0].astype(np.int16),slide_names=np.array(pd.factorize(idx.data_path)[1]),stained=(ab!='unstained'))
print(json.dumps(info,indent=0)); print('valid per target:',dict(zip(names,Mk.sum(0).tolist())))
print('cls3 counts',np.bincount(cls3[cls3>=0]).tolist(),'cls10',np.bincount(cls10[cls10>=0]).tolist())
# sanity: 3-class marker means on calibrated scale (batch-0 'all')
for k,n in enumerate(['lymph','gran','mono']):
    s=cls3==k; print(n, {g:round(float(T[s,8+i].mean()),2) for i,g in enumerate(G4)})
