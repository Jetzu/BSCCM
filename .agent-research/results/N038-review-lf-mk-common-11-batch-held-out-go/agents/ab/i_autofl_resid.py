"""N038 assumption breaker I: how much of the in-distribution vs held-out gap is a draw-specific unmixing artefact (label anti-correlated with
the authors' autofl estimate)? Per single target x draw (split-2 cells of the D1b comparison): r(v5 label, asinh autofl_unmixed); then score
the N031 ensemble, the n015 and r18 FOV models on the raw v5 label and on the label residualised on asinh(autofl_unmixed) (OLS fit on split-0
cells of that slide)."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt, sp, sl = T['Y'], T['mask'], T['batch'], T['split'], T['slide']; sm = pd.read_parquet(W + '/bsccm_sm.parquet')
af = np.arcsinh(sm['autofluor_single_antibody_model_unmixed'].values / 10)
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
ens = {}
for d, b in (('b10', 0), ('b01', 1)):
    A = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; ens[b] = pd.DataFrame(np.mean([x[0] for x in A], 0), index=A[0][1])
Pn, Gn = load('n015_sslinit_fov'); Pr, Gr = load('n024_r18_v5_fov'); pn = pd.DataFrame(Pn, index=Gn); pr = pd.DataFrame(Pr, index=Gr)
rows = []
for m in ['CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3']:
    j = names.index(f'single:{m}')
    for b in (0, 1):
        s = M[:, j] & (bt == b); fit = s & (sp == 0); A = np.c_[np.ones(fit.sum()), af[fit]]; coef = np.linalg.lstsq(A, Y[fit, j], rcond=None)[0]
        yres = Y[:, j] - (coef[0] + coef[1] * af)
        c = np.intersect1d(np.intersect1d(Gn[(bt[Gn] == b) & M[Gn, j]], ens[b].index.values), Gr)
        r = dict(target=m, draw=b, r_label_autofl=float(np.corrcoef(Y[s, j], af[s])[0, 1]))
        for nm, P in (('ens', ens[b]), ('n015', pn), ('r18', pr)):
            p = P.loc[c, j].values; r[f'{nm}_v5'] = float(np.corrcoef(p, Y[c, j])[0, 1]); r[f'{nm}_res'] = float(np.corrcoef(p, yres[c])[0, 1])
        rows.append(r)
d = pd.DataFrame(rows); d['gap_v5'] = d[['n015_v5', 'r18_v5']].max(axis=1) - d.ens_v5; d['gap_res'] = d[['n015_res', 'r18_res']].max(axis=1) - d.ens_res
pd.set_option('display.width', 250); print(d.round(3).to_string(index=False)); print(d[['ens_v5', 'ens_res', 'n015_v5', 'n015_res', 'r18_v5', 'r18_res', 'gap_v5', 'gap_res']].mean().round(4).to_dict())
d.to_csv(sys.argv[1] + '/i_autofl_resid.csv', index=False)
