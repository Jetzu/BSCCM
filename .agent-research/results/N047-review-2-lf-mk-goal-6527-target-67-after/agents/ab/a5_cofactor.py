import os; os.environ["OMP_NUM_THREADS"] = "4"
"""N047 AB (a5): is part of the CD19/CD56/CD3 FOV-offset 'label noise' created by OUR v3 transform asinh(x/q10) (q10 = slide 10th
percentile of the authors' unmixed value: tiny for dim stains -> near-log scale that magnifies additive background offsets of the
negative majority)? Per slide x draw: t_c = asinh(x/c) for cofactors c = q10 (v3), q50, q90, and fixed 50/200; replicate the v4 HGB position
detrend (fit on split 0), z-score; on split-2 test cells report FOV excess, r with the v5 label, and the 10-model ensemble's pooled and
within-FOV r against each version. A larger cofactor = labels-only re-base (NOT GOAL progress)."""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, M, bt, fov, sp, hr = T['Y'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
POS = ix[['fov_center_x', 'fov_center_y', 'fov_center_z', 'position_in_fov_x_pix', 'position_in_fov_y_pix']].values.astype(float)
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
ENS = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in fam]; ENS[b] = (np.mean([p['P'] for p in Ps], 0), Ps[0]['G'])
rng = np.random.default_rng(0)
def excess(x, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 2; x = x[ok]; inv = np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    sh = np.var((np.bincount(inv, x) / c)[inv]) / np.var(x); nul = np.mean([np.var((np.bincount(inv, xx) / c)[inv]) / np.var(xx) for xx in (rng.permutation(x) for _ in range(10))])
    return (sh - nul) / (1 - nul)
def wf(p, y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); return np.corrcoef(p - (np.bincount(inv, p) / c)[inv], y - (np.bincount(inv, y) / c)[inv])[0, 1]
rows = []
for m in ['CD19', 'CD56', 'CD3', 'HLA-DR', 'CD123', 'CD14', 'CD45', 'CD16']:
    j = names.index(f'single:{m}'); x = sm[f'{m}_single_antibody_model_unmixed'].values
    for b in (0, 1):
        s = np.flatnonzero((ab == m) & (bt == b) & M[:, j]); cal = s[sp[s] == 0]
        q = {p: np.percentile(x[cal], p) for p in (10, 50, 90)}
        P, G = ENS[b]; pos = pd.Series(np.arange(len(G)), index=G); te = s[(sp[s] == 2) & np.isin(s, G) & (hr[s] < 0)]; p = P[pos[te].values, j]
        for cn, c in (('q10(v3)', q[10]), ('q50', q[50]), ('q90', q[90]), ('50', 50.), ('200', 200.)):
            t = np.arcsinh(x[s] / c); tr = sp[s] == 0
            trend = HGB(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=0).fit(POS[s][tr], t[tr]).predict(POS[s])
            t = t - trend; t = (t - t[tr].mean()) / t[tr].std(); tt = pd.Series(t, index=s)[te].values
            rows.append(dict(target=m, draw=b, cofactor=cn, c=c, fov_excess=excess(tt, fov[te]), r_with_v5=np.corrcoef(tt, Y5[te, j])[0, 1],
                             ens_r=np.corrcoef(p, tt)[0, 1], ens_r_within=wf(p, tt, fov[te])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/a5_cofactor.csv', index=False); pd.set_option('display.width', 250)
print(df.round(3).to_string(index=False))
print(df.groupby('cofactor')[['fov_excess', 'ens_r', 'ens_r_within']].mean().round(4).to_string())
v5 = {(r.target, r.draw): r.ens_r for r in df[df.cofactor == 'q10(v3)'].itertuples()}
