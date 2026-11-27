"""N094 PRE-REGISTERED scorer (written 2026-11-27 before any seed-5 run finished). E8x2 = equal-weight mean of the 8 N085 E8 members,
each member averaged over two seeds: n048_v9_base (+ _seed5), n048_v9_zoom2 (+ _seed5), n053_cytoland (+ n053_cytoland_s1), N063
composite (seeds 2+3, as in E8), N065 composites z2 / cyto (+ _seed5 versions of both panel and twopart models), ConvNeXt-S plain
(s0+s1 as in E8), ConvNeXt-S composite (s0 + seed5). PRIMARY: GOAL(E8x2) > .7021 (= .6967 + 2 x .0027). Secondary: E8 (seed-5 versions
only, an independent replicate of E8), E8 (original), MM3. writes n094_scores.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r, d: np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(tp, d), L(pa, d); assert np.array_equal(a['G'], b['G']); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
def members(d, which):   # which: 'orig' (E8 as in N085), 'seed5' (replicate), 'both' (2-seed mean)
    plain = ('n085_cnxs120_plain_s0', 'n085_cnxs120_plain_s1') if d == 'b10' else ('n083_cnxs120_s0', 'n084_cnxs120_s1')
    o = {'base': L('n048_v9_base', d)['P'], 'zoom2': L('n048_v9_zoom2', d)['P'], 'cyto': L('n053_cytoland', d)['P'],
         'comp_base': np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0),
         'comp_z2': comp('n065_tpfov_z2', 'n065_panel_z2', d)[0], 'comp_cyto': comp('n065_tpfov_cyto', 'n065_panel_cyto', d)[0],
         'cnxs_plain': np.mean([L(r, d)['P'] for r in plain], 0), 'cnxs_comp': comp('n085_cnxs120_tpfov_s0', 'n085_cnxs120_panel_s0', d)[0]}
    s5 = {'base': L('n048_v9_base_seed5', d)['P'], 'zoom2': L('n048_v9_zoom2_seed5', d)['P'], 'cyto': L('n053_cytoland_s1', d)['P'],
          'comp_base': o['comp_base'], 'comp_z2': comp('n065_tpfov_z2_seed5', 'n065_panel_z2_seed5', d)[0],
          'comp_cyto': comp('n065_tpfov_cyto_seed5', 'n065_panel_cyto_seed5', d)[0], 'cnxs_plain': o['cnxs_plain'],
          'cnxs_comp': comp('n085_cnxs120_tpfov_seed5', 'n085_cnxs120_panel_seed5', d)[0]}
    G = L('n048_v9_base', d)['G']
    for r in ('n048_v9_base_seed5', 'n053_cytoland_s1', 'n065_tpfov_z2_seed5', 'n085_cnxs120_panel_seed5'): assert np.array_equal(L(r, d)['G'], G), r
    if which == 'orig': return o, G
    if which == 'seed5': return s5, G
    return {k: (o[k] + s5[k]) / 2 for k in o}, G
def score(P, G):
    r = {j: np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1] for j in c11}; return np.mean(list(r.values())), np.mean([r[j] for j in pooled])
rows = []
for k, which in (('E8x2 (PRIMARY)', 'both'), ('E8 seed-5 replicate', 'seed5'), ('E8 (N085 record)', 'orig')):
    g, mm = [], []
    for d in ('b10', 'b01'):
        m, G = members(d, which); a, b = score(np.mean(list(m.values()), 0), G); g.append(a); mm.append(b)
    rows.append(dict(set=k, b10=g[0], b01=g[1], GOAL=np.mean(g), MM3=np.mean(mm)))
df = pd.DataFrame(rows).round(4); df.to_csv('n094_scores.csv', index=False); print(df.to_string(index=False))
e = df[df.set == 'E8x2 (PRIMARY)'].GOAL.iloc[0]; print('PRIMARY GOAL(E8x2) = %.4f -> %s (bar .7021)' % (e, 'IMPROVED' if e > .7021 else 'FLAT/WORSE'))
