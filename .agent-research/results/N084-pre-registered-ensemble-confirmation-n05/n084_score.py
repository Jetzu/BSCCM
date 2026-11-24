"""N084 PRE-REGISTERED (written before the seed-1 b01 run finished): E = equal-weight mean of the N053 GOAL-best ensemble members
(n048_v9_base, n048_v9_zoom2, n053_cytoland) + ConvNeXt-S SEED 1 (b10: n083_cnxs60_s1 on the 60-ep draw-1 SSL encoder; b01:
n084_cnxs120_s1 on the 120-ep draw-0 SSL encoder). PRIMARY: GOAL(E) > .6916 (= .6862 + 2 x .0027) -> improved GOAL. MM3 reported.
writes n084_scores.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
grp = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def score(P, G):
    r = {j: np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1] for j in c11}; return np.mean(list(r.values())), np.mean([r[j] for j in grp])
cnx = {'b10': 'n083_cnxs60_s1_b10', 'b01': 'n084_cnxs120_s1_b01'}; ens = ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')
sets = {'control r50': lambda d: [f'n051_v10_base_{d}'], 'ConvNeXt-S seed 1': lambda d: [cnx[d]],
        'N053 ensemble': lambda d: [f'{r}_{d}' for r in ens], 'E (PRIMARY): N053 ensemble + ConvNeXt-S seed 1': lambda d: [f'{r}_{d}' for r in ens] + [cnx[d]]}
rows = []
for k, f in sets.items():
    g, mm = [], []
    for d in ('b10', 'b01'):
        Z = [L(r) for r in f(d)]; G = Z[0]['G']; a, b = score(np.mean([z['P'] for z in Z], 0), G); g.append(a); mm.append(b)
    rows.append(dict(set=k, b10=g[0], b01=g[1], GOAL=np.mean(g), MM3=np.mean(mm)))
df = pd.DataFrame(rows).round(4); df.to_csv('n084_scores.csv', index=False); print(df.to_string(index=False))
e = df[df.set.str.startswith('E (PRIMARY)')].GOAL.iloc[0]; print('PRIMARY GOAL(E) = %.4f -> %s (bar .6916)' % (e, 'IMPROVED' if e > .6916 else 'FLAT'))
