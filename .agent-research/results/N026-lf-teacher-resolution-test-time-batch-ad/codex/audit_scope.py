"""Read-only SSL lineage and AdaBN eligibility control."""
import torch,numpy as np
from pathlib import Path
W=Path('/hnvme/workspace/v181bb10-bsccm'); torch.set_num_threads(1)
s=torch.load(W/'runs/n007_ssl_aug/ckpt.pt',map_location='cpu',weights_only=False); T=np.load(W/'targets_v5.npz')
print('SSL config',s['config']); print('SSL rows',len(s['rows']),'epoch',s['epoch'])
for d in ('b10','b01'):
    E=np.load(W/'runs'/f'n023_v5_{d}'/'eval_preds.npz'); G=E['G']; print(d,'eval rows',len(G),'in SSL training',np.isin(G,s['rows']).sum())
M=T['mask']; has=(M.any(1)|(T['cls3']>=0))&(T['hist_row']<0)
for b in (0,1):
    c=(T['batch']==b)&(T['split']==0)&(T['hist_row']<0); print('ADABN eligibility',b,'no-label filter',c.sum(),'has-label filter',(c&has).sum(),'excluded',(c&~has).sum())
