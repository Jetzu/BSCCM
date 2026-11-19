"""N066 diagnosis: is the label error that the 6-way ensemble cannot explain predictable from LF-observable acquisition context
(FOV cell count = density, within-FOV position / radius, focus z)? Cross-draw test: fit a residual model g(context, pred) on the
OTHER draw's held-out residuals, apply to this draw; also FOV-mean residual share. writes n066_fovresid.csv"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, F = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
ix = pd.read_parquet(W + '/bsccm_index.parquet', columns=['position_in_fov_x_pix', 'position_in_fov_y_pix', 'fov_center_z']); assert len(ix) == len(Y)
cnt = np.bincount(F)[F].astype(float)                                   # cells detected in the same FOV
px, py = ix.position_in_fov_x_pix.values.astype(float), ix.position_in_fov_y_pix.values.astype(float)
cx, cy = np.median(px), np.median(py); rad = np.hypot(px - cx, py - cy); z = ix.fov_center_z.values
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(f'{tp}_{d}'), L(f'{pa}_{d}'); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
ens = {}
for d in ('b10', 'b01'):
    base = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    z2, G = comp('n065_tpfov_z2', 'n065_panel_z2', d); cyt, _ = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)
    ens[d] = (np.mean([base, z2, cyt] + [L(f'{r}_{d}')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')], 0), G)
def feats(G, p, zs):
    return np.c_[cnt[G], rad[G], px[G], py[G], zs, p]
rows = []
for d, o in (('b10', 'b01'), ('b01', 'b10')):
    (P, G), (Po, Go) = ens[d], ens[o]
    for j in c11:
        m, mo = M[G, j], M[Go, j]
        g, go = G[m], Go[mo]; p, y, po, yo = P[m, j], Y[g, j], Po[mo, j], Y[go, j]
        zt = (z[g] - np.median(z[g])); zo = (z[go] - np.median(z[go]))   # focus relative to the slide median (slides differ in absolute z)
        res, reso = y - p, yo - po
        r0 = np.corrcoef(p, y)[0, 1]
        # FOV-mean share of residual variance (within this draw; descriptive)
        fm = pd.Series(res).groupby(F[g]).transform('mean').values; share = np.var(fm) / np.var(res)
        rng = np.random.default_rng(0); null = np.mean([np.var(pd.Series(rng.permutation(res)).groupby(F[g]).transform('mean').values) / np.var(res) for _ in range(50)])
        h = HGB(max_iter=200, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=40).fit(feats(go, po, zo), reso)
        r1 = np.corrcoef(p + h.predict(feats(g, p, zt)), y)[0, 1]
        pc = {k: np.corrcoef(v, res)[0, 1] for k, v in (('count', cnt[g]), ('radius', rad[g]), ('z', zt))}
        rows.append(dict(direction=d, target=names[j], n=int(m.sum()), r_ens=r0, r_ctx_crossdraw=r1, delta=r1 - r0,
                         fov_resid_share=share, fov_share_null=null, **{f'res_r_{k}': v for k, v in pc.items()}))
df = pd.DataFrame(rows); df.to_csv('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/n066_fovresid.csv', index=False)
print(df.round(3).to_string(index=False))
for d in ('b10', 'b01'): print(d, 'r_ens %.4f  r_ctx %.4f' % tuple(df[df.direction == d][['r_ens', 'r_ctx_crossdraw']].mean()))
print('GOAL ens %.4f  ctx %.4f' % (df.r_ens.mean(), df.r_ctx_crossdraw.mean()))
