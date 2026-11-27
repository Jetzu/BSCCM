import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
for k in T.files:
    a=T[k]; print(k,a.shape,a.dtype, a[:3] if a.ndim==1 else '')
names=[str(n) for n in T['names']]; print(names)
import collections
b=T['batch']; s=T['split']; print('batch',collections.Counter(b.tolist())); print('split',collections.Counter(s.tolist()))
sl=T['slide']; print('nslides',len(np.unique(sl)), np.unique(sl)[:50])
z=np.load(W+'/runs/n048_v9_base_b01/eval_preds.npz'); print(z.files, z['P'].shape, z['G'][:5])
G=z['G']; print('b01 test batch', collections.Counter(b[G].tolist()), 'split',collections.Counter(s[G].tolist()))
print('censored', T['censored'].shape, T['censored'].dtype)
