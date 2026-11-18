"""N059 lit idea 3 (CPU falsifier): does a bright NEIGHBOUR bias a cell's label (background ring / bleed) in a way visible in the image?
For each GOAL cell: nearest other cell in the same FOV (pixel distance from position_in_fov), proximity w = exp(-d/40 px) (crop is
128 px); neighbour signal = ensemble PREDICTION of the neighbour for the same target. Partial r of the ensemble residual (y - p) with
w x neighbour_pred, controlling for p. Kill if |partial r| < .05 on the key targets. writes n059_neighbour.csv"""
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, FOV = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
idx = pd.read_parquet(W + '/bsccm_index.parquet', columns=['position_in_fov_x_pix', 'position_in_fov_y_pix']); XY = idx.values.astype(float)
rows = []
for d in ('b10', 'b01'):
    Ps = [np.load(f'{W}/runs/{r}_{d}/eval_preds.npz') for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')]; G = Ps[0]['G']; P = np.mean([q['P'] for q in Ps], 0)
    nd, nb = np.full(len(G), np.inf), np.full(len(G), -1)
    for f in np.unique(FOV[G]):
        ii = np.where(FOV[G] == f)[0]
        if len(ii) < 2: continue
        dd, kk = cKDTree(XY[G[ii]]).query(XY[G[ii]], k=2); nd[ii] = dd[:, 1]; nb[ii] = ii[kk[:, 1]]
    ok = nb >= 0; w = np.where(ok, np.exp(-nd / 40.0), 0.0)
    for j in c11:
        m = M[G, j] & ok
        if m.sum() < 200: continue
        y = Y[G[m], j]; p = P[m, j]; res = y - p; x = w[m] * P[nb[m], j]
        A = np.c_[np.ones(m.sum()), p]; rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]; rr = res - A @ np.linalg.lstsq(A, res, rcond=None)[0]
        rows.append(dict(direction=d, target=names[j], n=int(m.sum()), median_nn_dist=float(np.median(nd[m])), partial_r=float(np.corrcoef(rx, rr)[0, 1]),
                         r_resid_proximity=float(np.corrcoef(w[m], res)[0, 1])))
o = pd.DataFrame(rows); o.to_csv('n059_neighbour.csv', index=False); print(o.round(3).to_string(index=False)); print('mean |partial r|', round(o.partial_r.abs().mean(), 3))
