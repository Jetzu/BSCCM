import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v3c_v10.npz',allow_pickle=True) if __import__('os').path.exists(W+'/targets_v3c_v10.npz') else None
idx=pd.read_parquet(W+'/bsccm_index.parquet',columns=['antibodies','batch']); ab=idx.antibodies.values; b=idx.batch.values
x=pd.read_parquet(W+'/bsccm_sm.parquet',columns=['CD123/HLA-DR/CD14_full_model_unmixed'])['CD123/HLA-DR/CD14_full_model_unmixed'].values
V=np.load(W+'/targets.npz'); sp=V['split']; h=V['hist_row']
for bb in (0,1):
    u=(ab=='unstained')&(b==bb)&(sp==0)&(h<0); a=(ab=='all')&(b==bb)
    med=np.median(x[u]); print('draw',bb,'unstained median',med,'frac unstained == median',round(float((x[u]==med).mean()),3),'frac all-slide == median',round(float((x[a]==med).mean()),3),'all-slide q',np.round(np.percentile(x[a],[10,50,90]),2))
if T is not None:
    names=[str(n) for n in T['names']]; j=names.index('grp:CD123/HLA-DR/CD14'); print('v3c_v10 T range on all d0',np.round(np.percentile(T['T'][(ab=='all')&(b==0),j],[1,25,50,75,99]),2))
