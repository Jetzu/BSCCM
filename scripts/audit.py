import pandas as pd, numpy as np, json, sys
W='/hnvme/workspace/v181bb10-bsccm'
idx=pd.read_parquet(f'{W}/bsccm_index.parquet'); sm=pd.read_parquet(f'{W}/bsccm_sm.parquet')
out={}
gates=[c for c in sm.columns if c.startswith('selection_') and ('gated' in c or c=='selection_lymphocytes')]
G=sm[gates].astype(bool)
cnt=G.sum(1); out['gate_cells_any']=int((cnt>0).sum()); out['gate_overlap_cells']=int((cnt>1).sum())
out['gate_sizes']={c:int(G[c].sum()) for c in gates}
allb0=(idx.antibodies=='all')&(idx.batch==0)
out['all_b0']=int(allb0.sum()); out['all_b0_labeled']=int(((cnt>0)&allb0).sum())
print(json.dumps(out,indent=1))
# marker value stats on 'all' cells per batch
cols=['CD123/HLA-DR/CD14_full_model_unmixed','CD3/CD19/CD56_full_model_unmixed','CD45_full_model_unmixed','CD16_full_model_unmixed']
for b in (0,1):
    m=(idx.antibodies=='all')&(idx.batch==b)
    print('batch',b,'all-antibody cells',int(m.sum())); print(np.arcsinh(sm.loc[m,cols]/50).describe().loc[['mean','std','min','50%','max']].round(2).to_string())
m=idx.antibodies=='unstained'; print('unstained:'); print(np.arcsinh(sm.loc[m,cols]/50).describe().loc[['mean','std','50%']].round(2).to_string())
# correlation of 4 channels with gate classes in batch0 all
lab=pd.Series('none',index=idx.index)
for c in gates: lab[G[c]]=c.replace('selection_gated_','').replace('selection_','')
m=allb0&(lab!='none')
print(np.arcsinh(sm.loc[m,cols]/50).groupby(lab[m]).mean().round(2).to_string())
# per-imaging_date counts of modalities of interest
t=idx.groupby(['batch','antibodies','slide_replicate']).size().unstack(fill_value=0); print(t.to_string())
