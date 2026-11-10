import numpy as np, sys
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def sc(P,G): return np.mean([np.corrcoef(P[M[G,j],j],Y[G[M[G,j]],j])[0,1] for j in c11])
def load(r): E=np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'],E['G']
for d in sys.argv[1:]:
    for lab,runs in [('transductive base',[f'n023_v5_{d}']),('transductive zoom2',[f'n026_zoom2_{d}']),('transductive base+zoom2',[f'n023_v5_{d}',f'n026_zoom2_{d}']),
                     ('INDUCTIVE base',[f'n031_base_{d}']),('INDUCTIVE zoom2',[f'n031_zoom2_{d}']),('INDUCTIVE base+zoom2',[f'n031_base_{d}',f'n031_zoom2_{d}'])]:
        Ps=[load(r) for r in runs]; G=Ps[0][1]; assert all((p[1]==G).all() for p in Ps)
        print(f'{d} {lab:26s} {sc(np.mean([p[0] for p in Ps],0),G):.4f}')
