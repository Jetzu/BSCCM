import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; sl=T['slide']
J=[names.index(g) for g in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
# label-only QC statistic on ALL cells of each 'all' slide: among top-40% grp:CD16 cells (granulocyte-like, label-defined), corr(grp:CD3/19/56, grp:CD45)
for s in (8,14,13):
    k=(sl==s)&M[:,J].all(1); y=Y[k][:,J]; hi=y[:,2]>=np.quantile(y[:,2],.6)
    print('slide',s,'n',k.sum(),'label-only QC r(grp3,grp45 | grp16 top40%%) = %.2f'%np.corrcoef(y[hi,0],y[hi,1])[0,1])
sd=pd.read_csv('ab_per_slide.csv'); df=pd.read_csv('ab_oracles_per_target.csv')
b10=df[df.dir=='b10'].r.mean(); b01=df[df.dir=='b01'].set_index('target').r.copy()
rep1=sd[(sd.dir=='b01')&sd.slide.str.endswith('replicate_1')].set_index('target').r
b=b01.copy(); b['grp:CD45']=rep1['grp:CD45']; print('GOAL, grp:CD45 b01 scored on rep1 only: %.4f'%((b10+b.mean())/2))
b=b01.copy(); 
for t in rep1.index: b[t]=rep1[t]
print('GOAL, all pooled b01 on rep1 only: %.4f'%((b10+b.mean())/2))
