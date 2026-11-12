"""N038 assumption breaker B: label reliability of single-antibody targets (CD19/CD56/CD3/CD123/HLA-DR vs CD14/CD16/CD45), per draw.
All on the stained single-antibody slide (labels only, no model), clean cells, all splits.
 1. FOV variance decomposition of the v5 label: share of FOV means vs permutation null -> excess (FOV-offset) fraction f_fov.
 2. Technical control: the group full-model unmixed value (same dye dimension) on the UNSTAINED slide(s) of the same draw, transformed
    with the STAINED slide's q10 (same units): absolute FOV-level variance (unstained) / absolute FOV-level variance (stained raw v3-like).
 3. Same FOV decomposition for the authors' dye-channel shading_corrected value (background_lowess-subtracted) vs our fm_prep border-ring
    integral of the same channel (local per-cell background): which background model carries the FOV offsets?
 4. Spectral three-indicator reliability (CD3/CD19/CD56 dye: 500-550, 585-625, 627-673) on autofl-regressed residuals (autofl model
    fitted on unstained cells of the same draw from the 426-446 channel), pooled and within-FOV.
 5. 2-component GMM on the v5 label (pooled and FOV-demeaned): positive fraction, eta^2 (between-component variance share) ->
    sqrt(eta^2) = r of a perfect type-only predictor; within-negative variance share.
 6. Label ~ photons: FOV-grouped 5-fold CV of the v5 label from (a) fm_prep integ ridge (= N035), (b) authors' 6 shading_corrected
    channels ridge on asinh, (c) (b) + position HGB, (d) v3 label from (b) ridge."""
import os; os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys, json, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import cross_val_predict, GroupKFold
W = '/hnvme/workspace/v181bb10-bsccm'; OUT = sys.argv[1]
T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, Y3, M, bt, fov, sp, hr = T['Y'], T['Y_v3'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
F = np.load(W + '/fm_prep.npz', allow_pickle=True); I = F['integ'][:, :, 0]
CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']; SC = np.stack([sm[f'Fluor_{c}_shading_corrected'].values for c in CH], 1)
POS = ix[['fov_center_x', 'fov_center_y', 'fov_center_z', 'position_in_fov_x_pix', 'position_in_fov_y_pix']].values.astype(float)
GRP = {'CD19': 'CD3/CD19/CD56', 'CD56': 'CD3/CD19/CD56', 'CD3': 'CD3/CD19/CD56', 'CD123': 'CD123/HLA-DR/CD14', 'HLA-DR': 'CD123/HLA-DR/CD14',
       'CD14': 'CD123/HLA-DR/CD14', 'CD45': 'CD45', 'CD16': 'CD16'}
DYECH = {'CD3/CD19/CD56': 1, 'CD123/HLA-DR/CD14': 0, 'CD45': 0, 'CD16': 0}   # main dye channel index (N023 dye_spectra.txt)
rng = np.random.default_rng(0)
def fovstats(y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 3; y = y[ok]; inv = np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    mu = np.bincount(inv, y) / c; v_between = np.var(mu[inv]); share = v_between / np.var(y)
    nul = np.mean([np.var((np.bincount(inv, yy) / c)[inv]) / np.var(yy) for yy in (rng.permutation(y) for _ in range(30))])
    return dict(share=float(share), null=float(nul), excess=float((share - nul) / (1 - nul)), v_between=float(v_between), v_total=float(np.var(y)), cpf=float(c.mean()))
def demean(y, f):
    u, inv = np.unique(f, return_inverse=True); return y - (np.bincount(inv, y) / np.bincount(inv))[inv]
def gmm(y):
    g = GaussianMixture(2, random_state=0, n_init=3).fit(y[:, None]); mu = g.means_[:, 0]; w = g.weights_; var = g.covariances_.ravel(); k = np.argmax(mu)
    tot = np.var(y); eta2 = float(np.sum(w * (mu - np.sum(w * mu)) ** 2) / tot)
    return dict(pos_frac=float(w[k]), eta2=eta2, r_typeonly=float(np.sqrt(eta2)), dprime=float((mu[k] - mu[1 - k]) / np.sqrt(var[1 - k])),
                neg_within_share=float(w[1 - k] * var[1 - k] / tot), pos_within_share=float(w[k] * var[k] / tot))
def cvr(X, y, g, model):
    p = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=g); return float(np.corrcoef(p, y)[0, 1])
asin = lambda x, q: np.arcsinh(x / q)
rows = []
for m in ['CD19', 'CD56', 'CD3', 'CD123', 'HLA-DR', 'CD14', 'CD16', 'CD45']:
    j = names.index(f'single:{m}'); grp = GRP[m]; xg = sm[f'{grp}_full_model_unmixed'].values; xs = sm[f'{m}_single_antibody_model_unmixed'].values; dc = DYECH[grp]
    for b in (0, 1):
        s = (ab == m) & (bt == b) & M[:, j] & (hr < 0); u = (ab == 'unstained') & (bt == b) & (hr < 0)
        r = dict(target=m, draw=b, n=int(s.sum()))
        st5 = fovstats(Y5[s, j], fov[s]); r.update({f'v5_fov_{k}': v for k, v in st5.items() if k in ('share', 'null', 'excess', 'cpf')})
        st3 = fovstats(Y3[s, j], fov[s]); r.update(v3_fov_excess=st3['excess'])
        # technical control on unstained cells: same group unmixed column, stained slide's q10 units
        q = np.percentile(xg[s & (sp == 0)], 10); a_s = asin(xg[s], q); a_u = asin(xg[u], q)
        fs, fu = fovstats(a_s, fov[s]), fovstats(a_u, fov[u])
        r.update(grp_stained_fov_excess=fs['excess'], grp_unst_fov_excess=fu['excess'], unst_over_stained_vbetween=fu['v_between'] / fs['v_between'],
                 unst_over_stained_vtotal=fu['v_total'] / fs['v_total'])
        # authors' channel (background_lowess) vs our border-ring integral of the main dye channel
        fa, fm_ = fovstats(np.arcsinh(SC[s, dc] / np.percentile(np.abs(SC[s & (sp == 0), dc]), 50)), fov[s]), fovstats(np.arcsinh(I[s, dc] / np.percentile(np.abs(I[s & (sp == 0), dc]), 50)), fov[s])
        r.update(authors_ch_fov_excess=fa['excess'], ringbg_ch_fov_excess=fm_['excess'])
        # GMM type-only ceiling
        y = Y5[s, j]
        for tag, yy in (('pooled', y), ('fovdm', demean(y, fov[s]))):
            gm = gmm(yy); r.update({f'gmm_{tag}_{k}': v for k, v in gm.items()})
        # label ~ photons (FOV-grouped CV)
        Xi = (np.sign(F['integ'][s]) * np.log1p(np.abs(F['integ'][s]))).reshape(s.sum(), -1); Xa = np.arcsinh(SC[s] / 10.0)
        rid = RidgeCV(alphas=np.logspace(-2, 3, 8)); hgb = HGB(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, random_state=0)
        r['v5_from_fmprep_ridge'] = cvr(Xi, y, fov[s], rid); r['v5_from_authors6_ridge'] = cvr(Xa, y, fov[s], rid)
        r['v5_from_authors6_pos_hgb'] = cvr(np.c_[Xa, POS[s]], y, fov[s], hgb); r['v3_from_authors6_ridge'] = cvr(Xa, Y3[s, j], fov[s], rid)
        r['v3_from_fmprep_ridge'] = cvr(Xi, Y3[s, j], fov[s], rid)
        # spectral reliability for the CD3/CD19/CD56 dye: channels 500-550, 585-625, 627-673; autofl model from 426-446 (+690-) on unstained cells
        if grp == 'CD3/CD19/CD56':
            res = {}
            for c in (1, 3, 4):
                lr = LinearRegression().fit(SC[u][:, [0, 5]], SC[u, c]); res[c] = SC[s, c] - lr.predict(SC[s][:, [0, 5]])
            for tag, f_ in (('pooled', lambda z: z), ('withinFOV', lambda z: demean(z, fov[s]))):
                A, B_, C = (f_(res[c]) for c in (1, 3, 4)); rab, rac, rbc = np.corrcoef(A, B_)[0, 1], np.corrcoef(A, C)[0, 1], np.corrcoef(B_, C)[0, 1]
                r[f'spec_{tag}_rAB'] = rab; r[f'spec_{tag}_rAC'] = rac; r[f'spec_{tag}_rBC'] = rbc; r[f'spec_{tag}_relA'] = rab * rac / rbc
                # same on unstained cells: residual correlation = shared nuisance not removed by the autofl model
            Au, Bu = (SC[u, c] - LinearRegression().fit(SC[u][:, [0, 5]], SC[u, c]).predict(SC[u][:, [0, 5]]) for c in (1, 3))
            r['spec_unst_rAB'] = float(np.corrcoef(Au, Bu)[0, 1])
            # how much of the label is the spectral residual: r(v3 label, A residual) within slide
            r['r_v3_vs_resA'] = float(np.corrcoef(Y3[s, j], np.arcsinh(res[1] / 10))[0, 1])
        rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT + '/b_reliability.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(df[['target', 'draw', 'n', 'v5_fov_share', 'v5_fov_null', 'v5_fov_excess', 'v3_fov_excess', 'grp_stained_fov_excess', 'grp_unst_fov_excess', 'unst_over_stained_vbetween', 'unst_over_stained_vtotal', 'authors_ch_fov_excess', 'ringbg_ch_fov_excess']].round(3).to_string(index=False))
print(df[['target', 'draw', 'gmm_pooled_pos_frac', 'gmm_pooled_r_typeonly', 'gmm_pooled_dprime', 'gmm_pooled_neg_within_share', 'gmm_fovdm_pos_frac', 'gmm_fovdm_r_typeonly', 'gmm_fovdm_neg_within_share']].round(3).to_string(index=False))
print(df[['target', 'draw', 'v5_from_fmprep_ridge', 'v5_from_authors6_ridge', 'v5_from_authors6_pos_hgb', 'v3_from_authors6_ridge', 'v3_from_fmprep_ridge']].round(3).to_string(index=False))
sc = [c for c in df.columns if c.startswith('spec_') or c == 'r_v3_vs_resA']; print(df[df.target.isin(['CD19', 'CD56', 'CD3'])][['target', 'draw'] + sc].round(3).to_string(index=False))
