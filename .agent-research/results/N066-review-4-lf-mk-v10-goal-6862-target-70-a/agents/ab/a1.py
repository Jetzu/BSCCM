import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
print(list(T.keys()))
names=[str(n) for n in T['names']]; M=T['mask']; b=T['batch']; sl=T['slide']; sn=T['slide_names'].astype(str); sp=T['split']; h=T['hist_row']
print(names)
for bb in (0,1):
    for s in np.unique(sl[b==bb]):
        r=(sl==s)
        cnt={names[j]:int((M[r,j]&(h[r]<0)).sum()) for j in range(13) if M[r,j].any()}
        print(bb,sn[s],int(r.sum()),'split2',int((r&(sp==2)).sum()),cnt)
