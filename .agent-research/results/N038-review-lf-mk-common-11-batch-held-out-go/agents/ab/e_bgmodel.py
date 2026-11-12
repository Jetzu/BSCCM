"""N038 assumption breaker E: do the v5 label's FOV offsets come from the authors' smoothed background (background_lowess)?
Per single-antibody slide: (1) r(v5 label, dye-channel background error) where bgerr = (background - background_lowess)/shading in the main dye
channel, pooled and its FOV excess; (2) alternative label from the authors' per-cell LOCAL background: x_c = (total_raw - background)/shading,
least-squares unmixed like d_ringlabel (spectrum from the authors' gate, unstained mean + 2 PCs), asinh; scored with the N031 ensemble and
the n015 FOV model on split-2 test cells."""
import sys, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, M, bt, fov, sp, hr = T['Y'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
TOT = np.stack([sm[f'Fluor_{c}_total_raw'].values for c in CH], 1); BG = np.stack([sm[f'Fluor_{c}_background'].values for c in CH], 1)
BGL = np.stack([sm[f'Fluor_{c}_background_lowess'].values for c in CH], 1); SH = np.stack([sm[f'Fluor_{c}_shading'].values for c in CH], 1)
XL = (TOT - BG) / SH; XS = (TOT - BGL) / SH; BERR = (BG - BGL) / SH
DYE = {'CD19': 1, 'CD56': 1, 'CD3': 1, 'CD123': 0, 'HLA-DR': 0, 'CD14': 0, 'CD16': 0, 'CD45': 0}
rng = np.random.default_rng(0)
def fovx(y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 3; y = y[ok]; inv = np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    sh = np.var((np.bincount(inv, y) / c)[inv]) / np.var(y); nul = np.mean([np.var((np.bincount(inv, yy) / c)[inv]) / np.var(yy) for yy in (rng.permutation(y) for _ in range(20))])
    return float((sh - nul) / (1 - nul))
def wfov(p, y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); return float(np.corrcoef(p - (np.bincount(inv, p) / c)[inv], y - (np.bincount(inv, y) / c)[inv])[0, 1])
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
ens = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; ens[b] = (np.mean([p[0] for p in Ps], 0), Ps[0][1])
Pf, Gf = load('n015_sslinit_fov')
def at(P, G, rows): s = pd.Series(np.arange(len(G)), index=G); ok = np.isin(rows, G); return P[s[rows[ok]].values], ok
def unmix(X, s, u, g):
    spec = np.clip(X[s & g].mean(0) - X[s & ~g].mean(0), 0, None); spec /= np.linalg.norm(spec)
    mu = X[u].mean(0); pcs = np.linalg.svd(X[u] - mu, full_matrices=False)[2][:2]; B = np.stack([spec, mu / np.linalg.norm(mu), pcs[0], pcs[1]], 1)
    a = X @ np.linalg.pinv(B).T; a = a[:, 0]; return np.arcsinh(a / np.percentile(np.abs(a[s & (sp == 0)]), 50))
out = []
for m, dc in DYE.items():
    j = names.index(f'single:{m}')
    for b in (0, 1):
        s = (ab == m) & (bt == b) & M[:, j] & (hr < 0); u = (ab == 'unstained') & (bt == b) & (hr < 0); g = sm[f'selection_example_{m}_positive_cells_batch_{b}'].values.astype(bool)
        r = dict(target=m, draw=b)
        r['r_v5_bgerr_dye'] = float(np.corrcoef(Y5[s, j], BERR[s, dc])[0, 1]); r['r_v3_bgerr_dye'] = float(np.corrcoef(T['Y_v3'][s, j], BERR[s, dc])[0, 1])
        r['bgerr_fov_excess'] = fovx(BERR[s, dc], fov[s]); r['XS_dye_fov_excess'] = fovx(np.arcsinh(XS[s, dc] / 10), fov[s]); r['XL_dye_fov_excess'] = fovx(np.arcsinh(XL[s, dc] / 10), fov[s])
        lab = unmix(XL, s, u, g); r['loc_fov_excess'] = fovx(lab[s], fov[s]); r['v5_fov_excess'] = fovx(Y5[s, j], fov[s])
        gm = GaussianMixture(2, random_state=0, n_init=3).fit(lab[s][:, None]); k = np.argmax(gm.means_[:, 0]); w, mu_, var = gm.weights_, gm.means_[:, 0], gm.covariances_.ravel()
        r['loc_gmm_pos_frac'] = float(w[k]); r['loc_gmm_dprime'] = float((mu_[k] - mu_[1 - k]) / np.sqrt(var[1 - k]))
        te = np.flatnonzero(s & (sp == 2)); r['loc_vs_v5_test'] = float(np.corrcoef(lab[te], Y5[te, j])[0, 1])
        P, G = ens[b]; p, ok = at(P[:, j], G, te); tt = te[ok]
        r['ens_r_v5'] = float(np.corrcoef(p, Y5[tt, j])[0, 1]); r['ens_r_loc'] = float(np.corrcoef(p, lab[tt])[0, 1]); r['ens_rw_v5'] = wfov(p, Y5[tt, j], fov[tt]); r['ens_rw_loc'] = wfov(p, lab[tt], fov[tt])
        pf, okf = at(Pf[:, j], Gf, te); tf = te[okf]; r['fov_r_v5'] = float(np.corrcoef(pf, Y5[tf, j])[0, 1]); r['fov_r_loc'] = float(np.corrcoef(pf, lab[tf])[0, 1])
        out.append(r)
df = pd.DataFrame(out); df.to_csv(sys.argv[1] + '/e_bgmodel.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); print(df.round(3).to_string(index=False))
c11 = df  # 8 single targets only
print('mean over 16 single target x draw: ens r v5 %.4f  ens r loc %.4f | fov r v5 %.4f  fov r loc %.4f' % (df.ens_r_v5.mean(), df.ens_r_loc.mean(), df.fov_r_v5.mean(), df.fov_r_loc.mean()))
