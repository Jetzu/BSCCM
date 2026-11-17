"""N047 AB (a): are the FOV-level label offsets of the v5 GOAL labels image-visible / inductively learnable?
Per (target, test draw): residual e = z(y) - (a + b z(p)) of the 10-model inductive ensemble; FOV offset = FOV mean of e.
FOV features: LF-derived (log n cells, mean detection radius, FOV-mean ensemble prediction of all 13 outputs = composition),
imaging metadata (fov_center_z = focus), diagnostic-only (slide position x/y, acquisition order, fluorescence bg error).
Tests: (1) weighted corr of each feature with the offset (within test draw); (2) within-draw 2-fold FOV CV R2 (diagnostic, uses test labels);
(3) INDUCTIVE: fit offset ~ features on the OTHER draw (training draw for this direction), add predicted offset to p, re-score pooled r."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, fov, sl, bt = T['Y'], T['mask'], T['fov'], T['slide'], T['batch']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
ix = pd.read_parquet(W + '/bsccm_index.parquet'); sm = pd.read_parquet(W + '/bsccm_sm.parquet')
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
ENS = {}
for d in ('b10', 'b01'):
    Ps = [np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in fam]; ENS[d] = (np.mean([p['P'] for p in Ps], 0), Ps[0]['G'])
CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
BERR = np.stack([(sm[f'Fluor_{c}_background'].values - sm[f'Fluor_{c}_background_lowess'].values) / sm[f'Fluor_{c}_shading'].values for c in CH], 1)
# FOV-level features over ALL cells of the FOV
u, inv = np.unique(fov, return_inverse=True); cnt = np.bincount(inv)
def fmean(x): return (np.bincount(inv, x, minlength=len(u)) / cnt)
F = pd.DataFrame(index=u)
F['log_ncell'] = np.log(cnt); F['radius'] = fmean(ix.detection_radius.values.astype(float)); F['z_focus'] = fmean(ix.fov_center_z.values.astype(float))
F['pos_x'] = fmean(ix.fov_center_x.values.astype(float)); F['pos_y'] = fmean(ix.fov_center_y.values.astype(float)); F['slide'] = np.bincount(inv, sl, minlength=len(u)) // cnt
F['order'] = u.astype(float)  # global fov id ~ acquisition order (diagnostic)
for c in (0, 1): F[f'bgerr{c}'] = fmean(BERR[:, c])
# composition: FOV-mean ensemble prediction of each output over the FOV's eval cells (all eval cells of that direction)
for d, (P, G) in ENS.items():
    fi = inv[G]
    for j in range(13):
        s = np.bincount(fi, P[:, j], minlength=len(u)); c = np.bincount(fi, minlength=len(u)); v = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        F[f'comp{j}'] = np.where(np.isnan(v), F.get(f'comp{j}', np.nan), v)
LF_FEATS = ['log_ncell', 'radius'] + [f'comp{j}' for j in range(13)]
SETS = {'LF(count,size,composition)': LF_FEATS, 'LF+focus_z': LF_FEATS + ['z_focus'], 'DIAG position+order': ['pos_x', 'pos_y', 'order', 'z_focus'],
        'DIAG fluor bgerr': ['bgerr0', 'bgerr1']}
def zs(Xdf, slides):   # z-score features within slide (test-time, label-free)
    X = Xdf.values.astype(float).copy()
    for s in np.unique(slides):
        m = slides == s; mu = np.nanmean(X[m], 0); sd = np.nanstd(X[m], 0) + 1e-9; X[m] = (X[m] - mu) / sd
    return np.nan_to_num(X)
def offsets(d, j):
    P, G = ENS[d]; m = M[G, j]; g = G[m]; y = Y[g, j]; p = P[m, j]; y = (y - y.mean()) / y.std(); p = (p - p.mean()) / p.std()
    b, a = np.polyfit(p, y, 1); e = y - a - b * p; fi = inv[g]; uu, k = np.unique(fi, return_inverse=True); n = np.bincount(k); o = np.bincount(k, e) / n
    return dict(g=g, y=y, p=p, b=b, a=a, e=e, fidx=uu, k=k, n=n, o=o, fovs=u[uu])
other = {'b10': 'b01', 'b01': 'b10'}
rows = []; rng = np.random.default_rng(0)
for j in c11:
    for d in ('b10', 'b01'):
        R = offsets(d, j); Ro = offsets(other[d], j)
        sig2 = np.var(R['e']); keep = R['n'] >= 2
        # excess FOV variance of residual (offset) relative to noise sig2/n
        var_o = np.average(R['o'] ** 2, weights=R['n']); exp_noise = np.average(sig2 / R['n'], weights=R['n'])
        row = dict(target=names[j], dir=d, r_pool=np.corrcoef(R['p'], R['y'])[0, 1], n_fov=len(R['n']), offset_var_share=float((var_o - exp_noise) / np.var(R['y'])))
        Fx = F.iloc[R['fidx']]; Fo = F.iloc[Ro['fidx']]
        for fn in ['log_ncell', 'radius', 'z_focus', 'pos_x', 'pos_y', 'order', 'bgerr0', 'bgerr1'] + [f'comp{jj}' for jj in range(13)]:
            x = Fx[fn].values; ok = np.isfinite(x)
            if ok.sum() > 50: row[f'corr_{fn}'] = float(np.corrcoef(x[ok], R['o'][ok])[0, 1]) if np.std(x[ok]) > 0 else np.nan
        for sn, fs in SETS.items():
            Xt = zs(Fx[fs], Fx['slide'].values); Xo = zs(Fo[fs], Fo['slide'].values)
            # within-draw 2-fold FOV CV (diagnostic, uses test labels)
            fold = rng.integers(0, 2, len(R['o'])); pred = np.zeros(len(R['o']))
            for k in (0, 1):
                mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xt[fold != k], R['o'][fold != k], sample_weight=R['n'][fold != k]); pred[fold == k] = mdl.predict(Xt[fold == k])
            ss = np.average((R['o'] - pred) ** 2, weights=R['n']); s0 = np.average((R['o'] - np.average(R['o'], weights=R['n'])) ** 2, weights=R['n'])
            row[f'cvR2_{sn}'] = float(1 - ss / s0)
            pc = R['p'] + pred[R['k']] / R['b']; row[f'cv_r_{sn}'] = float(np.corrcoef(pc, R['y'])[0, 1])
            # inductive: fit on the other draw's FOV offsets (its labels = this direction's TRAINING draw), apply here
            mdl = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xo, Ro['o'], sample_weight=Ro['n']); pi = mdl.predict(Xt)
            ss = np.average((R['o'] - pi) ** 2, weights=R['n']); row[f'indR2_{sn}'] = float(1 - ss / s0)
            pc = R['p'] + pi[R['k']] / R['b']; row[f'ind_r_{sn}'] = float(np.corrcoef(pc, R['y'])[0, 1])
        # oracle: true leave-one-out FOV offset added (upper bound of a perfect FOV-offset predictor)
        loo = np.where(R['n'][R['k']] > 1, (R['o'][R['k']] * R['n'][R['k']] - R['e']) / np.maximum(R['n'][R['k']] - 1, 1), 0)
        row['oracle_loo_r'] = float(np.corrcoef(R['p'] + loo / R['b'], R['y'])[0, 1])
        rows.append(row)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/a_fov_offsets.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(df[['target', 'dir', 'r_pool', 'offset_var_share', 'oracle_loo_r'] + [c for c in df.columns if c.startswith('ind_r_') or c.startswith('indR2_') or c.startswith('cvR2_')]].round(3).to_string(index=False))
print(df[['target', 'dir'] + [c for c in df.columns if c.startswith('corr_')]].round(2).to_string(index=False))
g = lambda c: df.groupby('dir')[c].mean()
print('GOAL pooled', g('r_pool').round(4).to_dict(), 'mean', round(df.r_pool.mean(), 4))
for c in [c for c in df.columns if c.startswith('ind_r_') or c.startswith('cv_r_')] + ['oracle_loo_r']:
    print(f'{c:40s}', g(c).round(4).to_dict(), 'GOAL', round(df[c].mean(), 4), 'delta', round(df[c].mean() - df.r_pool.mean(), 4))
