"""N085 PRE-REGISTERED (written 2026-11-24 before any new N085 member finished). Union ensemble E8 = equal-weight mean of 8 members, each
of which passed a single-model screen:
  N053 members: n048_v9_base, n048_v9_zoom2, n053_cytoland (3);
  N065 composite members (singles from the twopart/fov-re model, pooled groups from the panel model): comp_base = mean of N063 seeds 2+3,
    comp_z2 (n065_*_z2), comp_cyto (n065_*_cyto) (3);
  ConvNeXt-S plain (120-ep in-domain SSL): b10 = mean(n085_cnxs120_plain_s0, _s1) on the new draw-1 encoder, b01 = mean(n083_cnxs120_s0,
    n084_cnxs120_s1) (1 member);
  ConvNeXt-S composite (120-ep SSL, seed 0): singles from n085_cnxs120_tpfov_s0, pooled from n085_cnxs120_panel_s0 (1 member).
PRIMARY: GOAL(E8) > .6916 (= GOAL best .6862 + 2 x .0027). Secondary (descriptive): MM3; E6 = first 6 members; each ConvNeXt-S member alone.
writes n085_scores.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r, d: np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(tp, d), L(pa, d); assert np.array_equal(a['G'], b['G']); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
def members(d):
    m = {}
    for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland'): z = L(r, d); m[r] = z['P']; G = z['G']
    m['comp_base'] = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    m['comp_z2'] = comp('n065_tpfov_z2', 'n065_panel_z2', d)[0]; m['comp_cyto'] = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)[0]
    plain = ('n085_cnxs120_plain_s0', 'n085_cnxs120_plain_s1') if d == 'b10' else ('n083_cnxs120_s0', 'n084_cnxs120_s1')
    m['cnxs_plain'] = np.mean([L(r, d)['P'] for r in plain], 0)
    m['cnxs_comp'] = comp('n085_cnxs120_tpfov_s0', 'n085_cnxs120_panel_s0', d)[0]
    for k in m: assert m[k].shape == m['n048_v9_base'].shape, k
    return m, G
def score(P, G):
    r = {j: np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1] for j in c11}; return np.mean(list(r.values())), np.mean([r[j] for j in pooled])
SETS = {'E8 (PRIMARY)': ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland', 'comp_base', 'comp_z2', 'comp_cyto', 'cnxs_plain', 'cnxs_comp'],
        'E6 (no ConvNeXt-S)': ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland', 'comp_base', 'comp_z2', 'comp_cyto'],
        'N053 ensemble': ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland'], 'cnxs_plain': ['cnxs_plain'], 'cnxs_comp': ['cnxs_comp']}
per = {d: members(d) for d in ('b10', 'b01')}; rows = []
for k, keys in SETS.items():
    g, mm = [], []
    for d in ('b10', 'b01'):
        m, G = per[d]; a, b = score(np.mean([m[x] for x in keys], 0), G); g.append(a); mm.append(b)
    rows.append(dict(set=k, b10=g[0], b01=g[1], GOAL=np.mean(g), MM3=np.mean(mm)))
df = pd.DataFrame(rows).round(4); df.to_csv('n085_scores.csv', index=False); print(df.to_string(index=False))
e = df[df.set == 'E8 (PRIMARY)'].GOAL.iloc[0]; print('PRIMARY GOAL(E8) = %.4f -> %s (bar .6916)' % (e, 'IMPROVED' if e > .6916 else 'FLAT'))
