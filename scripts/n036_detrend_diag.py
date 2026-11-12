"""N036 held-out diagnostics of the label position detrend: v3c (no detrend), v5c (v4 HGB fitted on split 0, in-sample for split-0
cells), v5cx (FOV-fold cross-fitted). Clean (non-Giemsa), masked cells only. Per target x blood draw x split:
  fov_share   = variance share of FOV means (cells in FOVs with >= 3 cells) -- an in-sample-overfit detrend makes split 0 lower than 1/2;
  pos_r_held  = residual position predictability on HELD-OUT FOVs (splits 1+2): per slide, FOV-grouped 5-fold CV HistGradientBoosting
                of the label on the 5 position features, out-of-fold Pearson r (median over slides) -- the artefact left in test labels.
Note v3c and v5c/v5cx differ in scale for multi-slide targets (v5 per-slide robust z), so compare shares/r, not raw variances.
Usage: n036_detrend_diag.py OUTDIR"""
import os; os.environ.setdefault('OMP_NUM_THREADS', '3')
import sys, json, numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W = '/hnvme/workspace/v181bb10-bsccm'; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
VERS = {'v3c': 'targets_v3c.npz', 'v5c': 'targets_v5c.npz', 'v5cx': 'targets_v5cx.npz'}
TT = {k: {kk: f[kk] for kk in ('Y', 'mask')} for k, v in VERS.items() for f in [np.load(W + '/' + v, allow_pickle=True)]}   # materialised: lazy NpzFile handles break in forked workers
T0 = np.load(W + '/targets_v5c.npz', allow_pickle=True); names = [str(n) for n in T0['names']]; fov, sl, sp, bt = (np.asarray(T0[k]) for k in ('fov', 'slide', 'split', 'batch')); clean = np.asarray(T0['hist_row']) < 0
idx = pd.read_parquet(W + '/bsccm_index.parquet')
P = idx[['fov_center_x', 'fov_center_y', 'fov_center_z', 'position_in_fov_x_pix', 'position_in_fov_y_pix']].values.astype(float)
FOLD = ((fov.astype(np.int64) * 2654435761) % 2**32) % 5
def share(y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 3
    if ok.sum() < 100: return np.nan
    y, inv = y[ok], np.unique(inv[ok], return_inverse=True)[1]; mu = np.bincount(inv, y) / np.bincount(inv); return float(np.var(mu[inv]) / np.var(y))
rows = []
for v, T in TT.items():
    for j, n in enumerate(names):
        for b in (0, 1):
            for s_ in (0, 1, 2):
                m = clean & T['mask'][:, j] & (bt == b) & (sp == s_)
                if m.sum() > 300: rows.append(dict(version=v, target=n, batch=b, split=s_, n=int(m.sum()), fov_share=share(T['Y'][m, j], fov[m])))
D1 = pd.DataFrame(rows); D1.to_csv(OUT + '/fov_share.csv', index=False)
def _pos(args):
    v, j, s = args; T = TT[v]
    m = np.flatnonzero(clean & T['mask'][:, j] & (sl == s) & (sp > 0))   # held-out FOVs only
    if len(m) < 500: return None
    y = T['Y'][m, j].astype(float); pred = np.zeros(len(m)); f = FOLD[m]
    for k in range(5):
        tr, te = f != k, f == k
        if te.sum() < 20 or tr.sum() < 200: return None
        pred[te] = HGB(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=0).fit(P[m][tr], y[tr]).predict(P[m][te])
    return dict(version=v, target=names[j], slide=int(s), batch=int(bt[m[0]]), n=int(len(m)), pos_r_held=float(np.corrcoef(pred, y)[0, 1]))
jobs = [(v, j, s) for v in VERS for j in range(len(names)) for s in np.unique(sl)]
with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 16)) // 3) as pool: D2 = pd.DataFrame([r for r in pool.map(_pos, jobs, chunksize=4) if r])
D2.to_csv(OUT + '/pos_r_heldout.csv', index=False)
pd.set_option('display.width', 220)
print(D1.pivot_table(index=['target', 'batch'], columns=['version', 'split'], values='fov_share').round(2).to_string())
print(D2.groupby(['target', 'version']).pos_r_held.median().unstack().round(3).to_string())
print('MEAN over targets/draws: fov_share', D1.groupby(['version', 'split']).fov_share.mean().unstack().round(3).to_string())
print('median pos_r_held', D2.groupby('version').pos_r_held.median().round(3).to_string())
