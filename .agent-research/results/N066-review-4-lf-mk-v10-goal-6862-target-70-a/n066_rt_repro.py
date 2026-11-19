"""N066: first-hand reproduction of the Codex diagnosis red-team (codex/diag_rt/diag_redteam.md): (i) noise-matched control L* = cY + e
(e orthogonal to Y and P) -> consensus r = a*sqrt((1+c)/2), computed from n066_altlabels.csv; (ii) add half of only the FOV-mean of
D = z(local) - z(v10) vs half of only the cell-centred D (evaluation-label diagnostics). writes n066_rt_repro.csv"""
import numpy as np, pandas as pd
alt = pd.read_csv('n066_altlabels.csv'); alt['r_nomatch_ctrl'] = alt.r_v10 * np.sqrt((1 + alt.r_v10_local) / 2)
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, F = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
H = np.load(W + '/runs/n064_ceiling/half_labels.npz'); hn = [str(n) for n in H['names']]; row = {g: i for i, g in enumerate(H['G'])}
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(f'{tp}_{d}'), L(f'{pa}_{d}'); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
out = []
for d in ('b10', 'b01'):
    base = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    z2, G = comp('n065_tpfov_z2', 'n065_panel_z2', d); cy, _ = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)
    P = np.mean([base, z2, cy] + [L(f'{r}_{d}')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')], 0)
    for j in c11:
        m = M[G, j]; g = G[m]; p = P[m, j]; y = Y[g, j]; yl = H['full'][[row[x] for x in g], hn.index(names[j])]
        z = lambda v: (v - v.mean()) / v.std(); D = z(yl) - z(y); fm = pd.Series(D).groupby(F[g]).transform('mean').values
        out.append(dict(direction=d, target=names[j], r_v10=np.corrcoef(p, y)[0, 1], r_fovhalf=np.corrcoef(p, z(y) + fm / 2)[0, 1], r_cellhalf=np.corrcoef(p, z(y) + (D - fm) / 2)[0, 1]))
o = pd.DataFrame(out).merge(alt[['direction', 'target', 'r_consensus', 'r_nomatch_ctrl']], on=['direction', 'target'])
o.to_csv('n066_rt_repro.csv', index=False); print(o.round(3).to_string(index=False)); print(o.drop(columns=['direction', 'target']).mean().round(4).to_string())
