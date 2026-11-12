"""N038 assumption breaker D: is the GOAL predictor better than the v5 label lets it show? Build an independent 'ring-background' label
from our own fluorescence integrals (fm_prep: per-cell border-ring median background, aperture r<32 = annuli 1+2), unmixed per cell by
least squares on [dye spectrum, unstained mean, unstained PC1, unstained PC2] (6 channels). Dye spectrum = mean integral of the
authors' spectrum-estimation gate (selection_example_<m>_positive_cells_batch_<b>) minus the rest of the slide (direction only).
Score saved predictions (N031 ensemble, batch-held-out; n015 FOV model = in-distribution) on split-2 test cells against v5 and ring
labels: pooled r, within-FOV r, FOV excess of each label, GMM type-only ceiling of the ring label."""
import sys, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, M, bt, fov, sp, hr = T['Y'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
F = np.load(W + '/fm_prep.npz', allow_pickle=True); IA = F['integ'][:, :, 1] + F['integ'][:, :, 2]   # r < 32 aperture, ring-bg subtracted
rng = np.random.default_rng(0)
def fovx(y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 3; y = y[ok]; inv = np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    sh = np.var((np.bincount(inv, y) / c)[inv]) / np.var(y); nul = np.mean([np.var((np.bincount(inv, yy) / c)[inv]) / np.var(yy) for yy in (rng.permutation(y) for _ in range(20))])
    return (sh - nul) / (1 - nul)
def wfov(p, y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv)
    return np.corrcoef(p - (np.bincount(inv, p) / c)[inv], y - (np.bincount(inv, y) / c)[inv])[0, 1]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
ens = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; ens[b] = (np.mean([p[0] for p in Ps], 0), Ps[0][1])
Pf, Gf = load('n015_sslinit_fov')
def at(P, G, rows): s = pd.Series(np.arange(len(G)), index=G); ok = np.isin(rows, G); return P[s[rows[ok]].values], ok
out = []
for m in ['CD19', 'CD56', 'CD3', 'CD123', 'HLA-DR', 'CD14', 'CD16', 'CD45']:
    j = names.index(f'single:{m}')
    for b in (0, 1):
        s = (ab == m) & (bt == b) & M[:, j] & (hr < 0); u = (ab == 'unstained') & (bt == b) & (hr < 0)
        g = sm[f'selection_example_{m}_positive_cells_batch_{b}'].values.astype(bool)
        spec = IA[s & g].mean(0) - IA[s & ~g].mean(0); spec = np.clip(spec, 0, None); spec /= np.linalg.norm(spec)
        Xu = IA[u]; mu = Xu.mean(0); pcs = np.linalg.svd(Xu - mu, full_matrices=False)[2][:2]
        B = np.stack([spec, mu / np.linalg.norm(mu), pcs[0], pcs[1]], 1)   # 6 x 4
        coef = IA @ np.linalg.pinv(B).T; a = coef[:, 0]
        q = np.percentile(np.abs(a[s & (sp == 0)]), 50); ring = np.arcsinh(a / q)
        te = np.flatnonzero(s & (sp == 2)); r = dict(target=m, draw=b, n_test=len(te), spec=np.round(spec, 2).tolist())
        r['ring_vs_v5_test'] = float(np.corrcoef(ring[te], Y5[te, j])[0, 1])
        r['v5_fov_excess'] = fovx(Y5[s, j], fov[s]); r['ring_fov_excess'] = fovx(ring[s], fov[s])
        gm = GaussianMixture(2, random_state=0, n_init=3).fit(ring[s][:, None]); k = np.argmax(gm.means_[:, 0]); w, mu_, var = gm.weights_, gm.means_[:, 0], gm.covariances_.ravel()
        r['ring_gmm_pos_frac'] = float(w[k]); r['ring_gmm_r_typeonly'] = float(np.sqrt(np.sum(w * (mu_ - np.sum(w * mu_)) ** 2) / np.var(ring[s]))); r['ring_gmm_dprime'] = float((mu_[k] - mu_[1 - k]) / np.sqrt(var[1 - k]))
        P, G = ens[b]; p, ok = at(P[:, j], G, te); tt = te[ok]
        r['ens_r_v5'] = float(np.corrcoef(p, Y5[tt, j])[0, 1]); r['ens_r_ring'] = float(np.corrcoef(p, ring[tt])[0, 1])
        r['ens_rw_v5'] = float(wfov(p, Y5[tt, j], fov[tt])); r['ens_rw_ring'] = float(wfov(p, ring[tt], fov[tt]))
        from scipy.stats import spearmanr
        r['ens_rho_v5'] = float(spearmanr(p, Y5[tt, j])[0]); r['ens_rho_ring'] = float(spearmanr(p, ring[tt])[0])
        pf, okf = at(Pf[:, j], Gf, te); tf = te[okf]
        r['fov_r_v5'] = float(np.corrcoef(pf, Y5[tf, j])[0, 1]); r['fov_r_ring'] = float(np.corrcoef(pf, ring[tf])[0, 1]); r['fov_rw_v5'] = float(wfov(pf, Y5[tf, j], fov[tf]))
        out.append(r); print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.items()}, flush=True)
df = pd.DataFrame(out); df.to_csv(sys.argv[1] + '/d_ringlabel.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df.drop(columns=['spec']).round(3).to_string(index=False))
