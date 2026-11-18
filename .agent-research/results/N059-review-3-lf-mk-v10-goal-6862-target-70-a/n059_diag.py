"""N059 diagnosis: where does the gap to .70 live? Best ensemble (N048 base + zoom2 + Cytoland, N053) vs in-distribution model
(n039_ceil_fov_v5_mt0: FOV split, both draws, v5-trained) on the SAME GOAL cells, v10 labels. Per target x direction:
pooled r (GOAL), within-slide r, within-FOV r (label FOV-level error removed), oracle per-slide affine recalibration (uses test labels:
calibration ceiling), Spearman. writes n059_diag.csv + prints direction/target means."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True)
Y, M, FOV, SL = T['Y'], T['mask'], T['fov'], T['slide']; names = [str(n) for n in T['names']]
c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
L = lambda r, d: np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')
def demean(x, g):
    s = pd.Series(x).groupby(g).transform('mean').values; return x - s
def stats(p, y, fov, sl):
    r = np.corrcoef(p, y)[0, 1]
    rs = np.corrcoef(demean(p, sl), demean(y, sl))[0, 1]
    rf = np.corrcoef(demean(p, fov), demean(y, fov))[0, 1]
    pa = p.copy()
    for s in np.unique(sl):
        m = sl == s
        if m.sum() > 10: A = np.c_[p[m], np.ones(m.sum())]; pa[m] = A @ np.linalg.lstsq(A, y[m], rcond=None)[0]
    ra = np.corrcoef(pa, y)[0, 1]; sp = spearmanr(p, y).correlation
    return r, rs, rf, ra, sp
ID = np.load(f'{W}/runs/n039_ceil_fov_v5_mt0/eval_preds.npz'); idmap = {g: i for i, g in enumerate(ID['G'])}
rows = []
for d in ('b10', 'b01'):
    Ps = [L(r, d) for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')]; G = Ps[0]['G']; P = np.mean([q['P'] for q in Ps], 0)
    for j in c11:
        m = M[G, j]; g = G[m]; y = Y[g, j]
        e = stats(P[m, j], y, FOV[g], SL[g])
        sel = np.array([x in idmap for x in g]); gi = np.array([idmap[x] for x in g[sel]])
        i = stats(ID['P'][gi, j], y[sel], FOV[g][sel], SL[g][sel]) if sel.sum() > 50 else (np.nan,) * 5
        rows.append(dict(direction=d, target=names[j], n=int(m.sum()), n_slides=len(np.unique(SL[g])),
                         ens_r=e[0], ens_within_slide=e[1], ens_within_fov=e[2], ens_oracle_slide_affine=e[3], ens_spearman=e[4],
                         indist_r=i[0], indist_within_fov=i[2], indist_oracle_slide_affine=i[3], indist_coverage=float(sel.mean())))
df = pd.DataFrame(rows); df.to_csv('n059_diag.csv', index=False)
cols = ['ens_r', 'ens_within_slide', 'ens_within_fov', 'ens_oracle_slide_affine', 'ens_spearman', 'indist_r', 'indist_within_fov', 'indist_oracle_slide_affine']
print(df.groupby('direction')[cols].mean().round(4).to_string()); print('GOAL-style mean over directions:'); print(df[cols].mean().round(4).to_string())
print(df[['direction', 'target', 'n_slides', 'ens_r', 'ens_within_fov', 'ens_oracle_slide_affine', 'indist_r']].round(3).to_string(index=False))
