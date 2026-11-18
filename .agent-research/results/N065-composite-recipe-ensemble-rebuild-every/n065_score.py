"""N065: composite-recipe ensemble (pre-registered). Composite member = singles (+autofl) from the twopart+fov-re model, pooled groups
(grp:CD3/CD19/CD56, grp:CD45, grp:CD16) from the panel model. Members: composite base (N063 seeds 2 and 3 averaged), composite zoom2
(n065_*_z2), composite Cytoland (n065_*_cyto). PRIMARY: E3c = mean of the 3 members; success if GOAL > .6862 + .0054 = .6916.
Secondary (descriptive): each member alone; E3c + N053 members (N048 base, N048 zoom2, N053 Cytoland) = 6-way.
writes n065_scores.csv, n065_per_target.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def rs(P, G): return {names[j]: float(np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1]) for j in c11}
def comp(tp, pa, d):
    a, b = L(f'{tp}_{d}'), L(f'{pa}_{d}'); assert np.array_equal(a['G'], b['G']); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
def members(d):
    base = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    z2, G = comp('n065_tpfov_z2', 'n065_panel_z2', d); cy, G2 = comp('n065_tpfov_cyto', 'n065_panel_cyto', d); assert np.array_equal(G, G2)
    old = [L(f'{r}_{d}')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')]
    return dict(comp_base=base, comp_z2=z2, comp_cyto=cy), old, G
sets = {'comp_base': ['comp_base'], 'comp_z2': ['comp_z2'], 'comp_cyto': ['comp_cyto'], 'E3c (PRIMARY)': ['comp_base', 'comp_z2', 'comp_cyto']}
rows, pt = [], []
per = {d: members(d) for d in ('b10', 'b01')}
for lab, keys in list(sets.items()) + [('E3c + N053 (6-way)', None)]:
    g = []
    for d in ('b10', 'b01'):
        mem, old, G = per[d]
        P = np.mean([mem[k] for k in keys], 0) if keys else np.mean([mem[k] for k in ('comp_base', 'comp_z2', 'comp_cyto')] + old, 0)
        r = rs(P, G); g.append(np.mean(list(r.values())))
        for k, v in r.items(): pt.append(dict(set=lab, direction=d, target=k, r=v))
        rows.append(dict(set=lab, direction=d, r=g[-1]))
    rows.append(dict(set=lab, direction='GOAL', r=float(np.mean(g))))
df = pd.DataFrame(rows); df.to_csv('n065_scores.csv', index=False); pd.DataFrame(pt).to_csv('n065_per_target.csv', index=False)
print(df.pivot(index='set', columns='direction', values='r').round(4).to_string())
e = df[(df.set == 'E3c (PRIMARY)') & (df.direction == 'GOAL')].r.iloc[0]
print('PRIMARY E3c = %.4f -> %s (bar .6916)' % (e, 'SUCCESS' if e > .6916 else 'FLAT'))
