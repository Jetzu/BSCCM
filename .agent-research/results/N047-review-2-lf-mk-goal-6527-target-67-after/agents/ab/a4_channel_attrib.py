import os; os.environ["OMP_NUM_THREADS"] = "4"
"""N047 AB (a4): WHICH channel's background error creates the v5 FOV offsets of CD19/CD56/CD3? Per single-antibody slide x draw: fit the
authors' unmixing as an affine map of the 6 lowess-background channels XS=(total-bg_lowess)/shading -> authors' unmixed value (lstsq on
all cells; fidelity printed); then swap in the PER-CELL background for one channel at a time (and for all), apply the same map, asinh(x/50),
replicate the v4 position detrend (HGB on fov centre x/y/z + position in fov, fitted on split-0 cells), and measure the FOV excess
(over a permutation null) on the split-2 test cells. Also the same for the authors' autofluorescence unmixed value."""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, M, bt, fov, sp, hr = T['Y'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
POS = ix[['fov_center_x', 'fov_center_y', 'fov_center_z', 'position_in_fov_x_pix', 'position_in_fov_y_pix']].values.astype(float)
CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
TOT = np.stack([sm[f'Fluor_{c}_total_raw'].values for c in CH], 1); BG = np.stack([sm[f'Fluor_{c}_background'].values for c in CH], 1)
BGL = np.stack([sm[f'Fluor_{c}_background_lowess'].values for c in CH], 1); SH = np.stack([sm[f'Fluor_{c}_shading'].values for c in CH], 1)
XS = (TOT - BGL) / SH; XL = (TOT - BG) / SH
rng = np.random.default_rng(0)
def excess(x, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 2; x = x[ok]; inv = np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    sh = np.var((np.bincount(inv, x) / c)[inv]) / np.var(x); nul = np.mean([np.var((np.bincount(inv, xx) / c)[inv]) / np.var(xx) for xx in (rng.permutation(x) for _ in range(10))])
    return (sh - nul) / (1 - nul)
def detrend(v, s):
    tr = s[sp[s] == 0]; mdl = HGB(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=0).fit(POS[tr], v[np.searchsorted(s, tr)])
    return v - mdl.predict(POS[s])
rows = []
for m in ['CD19', 'CD56', 'CD3']:
    j = names.index(f'single:{m}'); un = sm[f'{m}_single_antibody_model_unmixed'].values; af = sm['autofluor_single_antibody_model_unmixed'].values
    for b in (0, 1):
        s = np.flatnonzero((ab == m) & (bt == b) & (hr < 0) & np.isfinite(un)); te = sp[s] == 2
        for tgt_name, tv in (('marker', un),):
            ok = np.isfinite(tv[s]); X1 = np.c_[XS[s], np.ones(len(s))]; beta = np.linalg.lstsq(X1[ok], tv[s][ok], rcond=None)[0]
            fid = np.corrcoef(X1[ok] @ beta, tv[s][ok])[0, 1]
            r = dict(target=m, draw=b, quantity=tgt_name, fidelity=fid)
            r['coef'] = ' '.join(f'{c}:{v:+.2f}' for c, v in zip(CH, beta[:6]))
            variants = {'authors': tv[s], 'refit_lowess': X1 @ beta, 'percell_all': np.c_[XL[s], np.ones(len(s))] @ beta}
            for c in range(6):
                Xv = XS[s].copy(); Xv[:, c] = XL[s, c]; variants[f'percell_{CH[c]}'] = np.c_[Xv, np.ones(len(s))] @ beta
            for k, v in variants.items():
                v = np.arcsinh(np.nan_to_num(v) / 50); vd = detrend(v, s); r[f'exc_{k}'] = excess(vd[te], fov[s][te])
            if tgt_name == 'marker': r['exc_v5_label'] = excess(Y5[s[te], j], fov[s][te]) if M[s[te], j].all() else np.nan
            rows.append(r)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/a4_channel_attrib.csv', index=False); pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
print(df.drop(columns=['coef']).round(3).to_string(index=False)); print(df[['target', 'draw', 'quantity', 'coef']].to_string(index=False))
