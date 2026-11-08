"""Build targets.npz / cells.parquet: marker targets+masks, gate classes, FOV-level split ids, histology row map."""
import pandas as pd, numpy as np, json
W='/hnvme/workspace/v181bb10-bsccm'; COF=50.0
idx=pd.read_parquet(f'{W}/bsccm_index.parquet'); sm=pd.read_parquet(f'{W}/bsccm_sm.parquet'); N=len(idx)
ab=idx.antibodies.values
G4={'CD123/HLA-DR/CD14':{'CD123','HLA-DR','CD14'},'CD3/CD19/CD56':{'CD3','CD19','CD56'},'CD45':{'CD45'},'CD16':{'CD16'}}
S8=['CD45','CD123','CD19','CD56','CD14','CD16','HLA-DR','CD3']
names=[f'grp:{k}' for k in G4]+[f'single:{k}' for k in S8]
Y=np.zeros((N,len(names)),np.float32); Mk=np.zeros((N,len(names)),bool)
for j,(k,s) in enumerate(G4.items()):
    Y[:,j]=np.arcsinh(sm[f'{k}_full_model_unmixed'].values/COF); Mk[:,j]=np.isin(ab,list(s))|(ab=='all')
for j,k in enumerate(S8):
    jj=4+j; v=sm[f'{k}_single_antibody_model_unmixed'].values; Mk[:,jj]=(ab==k); Y[:,jj]=np.arcsinh(np.nan_to_num(v)/COF)
# gate classes (10 non-lymphocyte gates); exclusivity check
gates=[c for c in sm.columns if c.startswith('selection_gated_')]
Gm=sm[gates].astype(bool).values; ncls=Gm.sum(1)
print('gate exclusivity: cells with >1 of the 10 gates:',int((ncls>1).sum()),' with exactly 1:',int((ncls==1).sum()))
cls=np.where(ncls==1,Gm.argmax(1),-1).astype(np.int16)
# FOV id and split
fov=idx.groupby(['data_path','fov_center_x','fov_center_y']).ngroup().values
rng=np.random.default_rng(0); u=np.unique(fov); perm=rng.permutation(u); r=np.empty(len(u),int); r[perm]=np.arange(len(u))
frac=r[np.searchsorted(u,fov)]/len(u)
split=np.where(frac<0.8,0,np.where(frac<0.9,1,2)).astype(np.int8)   # 0 train 1 val 2 test, by FOV
print('n FOV',len(u)); print(pd.crosstab(split,idx.batch).to_string())
hist=np.load(f'{W}/pack/hist_cells.npy'); hrow=np.full(N,-1,np.int32); hrow[hist]=np.arange(len(hist))
np.savez(f'{W}/targets.npz',Y=Y,mask=Mk,names=np.array(names),cls=cls,gates=np.array([g.replace('selection_gated_','') for g in gates]),fov=fov,split=split,hist_row=hrow,batch=idx.batch.values,cof=COF)
print('targets valid counts:',dict(zip(names,Mk.sum(0).tolist()))); print('class counts:',np.bincount(cls[cls>=0]).tolist())
print('labeled & split:',pd.crosstab(cls>=0,split).to_string()); print('hist cells by split',np.bincount(split[hist]).tolist())
