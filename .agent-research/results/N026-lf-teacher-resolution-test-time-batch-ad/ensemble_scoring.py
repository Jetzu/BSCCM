import numpy as np, json, csv
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def sc(P,G): return np.mean([np.corrcoef(P[M[G,j],j],Y[G[M[G,j]],j])[0,1] for j in c11])
def load(r): E=np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'],E['G']
rows=[]; out={}
arms={'single_base':[('n023_v5_{d}',),('n023_v5_{d}_s1',)],
      'single_zoom2':[('n026_zoom2_{d}',),('n026_zoom2_{d}_s1',)],
      'single_dual2':[('n026_dual2_{d}',)],
      'ens_base+base':[('n023_v5_{d}','n023_v5_{d}_s1')],
      'ens_base+zoom2':[('n023_v5_{d}','n026_zoom2_{d}'),('n023_v5_{d}_s1','n026_zoom2_{d}_s1')]}
for arm,reps in arms.items():
    for s,combo in enumerate(reps):
        v={}
        for d in ('b10','b01'):
            Ps=[load(c.format(d=d)) for c in combo]; G=Ps[0][1]; assert all((p[1]==G).all() for p in Ps)
            v[d]=sc(np.mean([p[0] for p in Ps],0),G)
        g=(v['b10']+v['b01'])/2; rows.append(dict(arm=arm,seed=s,value=g)); print(f'{arm:16s} rep{s}: b10 {v["b10"]:.4f} b01 {v["b01"]:.4f} GOAL {g:.4f}')
with open('/home/hpc/v181bb/v181bb10/BSCCM/'+__import__('glob').glob('.agent-research/results/N026-*')[0]+'/goal_arms.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=['arm','seed','value']); w.writeheader(); w.writerows(rows)
