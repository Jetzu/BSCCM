"""N078 descriptive: MM3 (3 pooled panels) per arm, and the N053 GOAL-best ensemble (n048_v9_base + n048_v9_zoom2 + n053_cytoland)
with the ConvNeXt-S (cnxs_60) or r50_160 member added (equal-weight mean). writes n078_mm3_ens.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
grp = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r, d: np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')
def score(P, G):
    r = {j: np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1] for j in c11}; return np.mean(list(r.values())), np.mean([r[j] for j in grp])
rows = []
sets = {'control': ['n051_v10_base'], 'r50_160': ['n078_ft_r50_160_s0'], 'cnxs_60': ['n078_ft_cnxs_60_s0'],
        'N053 ens (best)': ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland'],
        'N053 ens + cnxs_60': ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland', 'n078_ft_cnxs_60_s0'],
        'N053 ens + r50_160': ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland', 'n078_ft_r50_160_s0']}
for k, runs in sets.items():
    g, mm = [], []
    for d in ('b10', 'b01'):
        Z = [L(r, d) for r in runs]; G = Z[0]['G']; assert all(np.array_equal(z['G'], G) for z in Z)
        a, b = score(np.mean([z['P'] for z in Z], 0), G); g.append(a); mm.append(b)
    rows.append(dict(set=k, GOAL=np.mean(g), b10=g[0], b01=g[1], MM3=np.mean(mm)))
df = pd.DataFrame(rows).round(4); df.to_csv('n078_mm3_ens.csv', index=False); print(df.to_string(index=False))
