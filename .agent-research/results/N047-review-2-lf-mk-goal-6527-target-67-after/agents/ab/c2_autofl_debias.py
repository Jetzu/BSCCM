"""N047 AB (c2): INDUCTIVE test of the draw-specific autofluorescence artefact in single-antibody labels. Training-draw labels contain
y = marker + c_train * AF (AF = authors' autofluor unmixed, asinh/50); the model learns c_train * E[AF|x]. Estimate c_train on the TRAINING
draw only (within-lymphocyte OLS slope of y on AF, authors' lymphocyte selection; also all-cell slope), map AF to the model's autofl head on
training cells (AF ~ a + b * autofl label), and remove the learned artefact from test predictions: p' = p - c_train * AF_hat.
No test-draw label, statistic or image is used. Also: oracle slope sweep on the test draw (lam * z(autofl head)) to show the sign flip."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt, hr = T['Y'], T['mask'], T['batch'], T['hist_row']; jaf = names.index('autofl')
sm = pd.read_parquet(W + '/bsccm_sm.parquet', columns=['autofluor_single_antibody_model_unmixed', 'selection_lymphocytes'])
ix = pd.read_parquet(W + '/bsccm_index.parquet', columns=['antibodies']); ab = ix.antibodies.values.astype(str)
AF = np.arcsinh(sm['autofluor_single_antibody_model_unmixed'].values / 50)
_E = np.load(W + '/runs/n039_ceil_fov_v5_mt128000/eval_preds.npz'); CL = np.full(len(Y), -1); CL[_E['G']] = _E['C3']   # label-free LF class (split-2 cells only)
def within_slope(x, y, c):   # pooled within-class slope (class fixed effects)
    xs = x.copy(); ys = y.copy()
    for k in np.unique(c): s = c == k; xs[s] -= xs[s].mean(); ys[s] -= ys[s].mean()
    return float((xs * ys).sum() / (xs * xs).sum())
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
rows = []
for d, te_b in (('b10', 0), ('b01', 1)):
    tr_b = 1 - te_b; Ps = [load(f'{m}_{d}') for m in fam]; P = np.mean([p[0] for p in Ps], 0); G = Ps[0][1]
    for mk in ['HLA-DR', 'CD19', 'CD14', 'CD123', 'CD56', 'CD3', 'CD45', 'CD16']:
        j = names.index(f'single:{mk}')
        trc = np.flatnonzero((ab == mk) & (bt == tr_b) & M[:, j] & (hr < 0) & np.isfinite(AF))
        t2 = trc[CL[trc] >= 0]; c_ly = within_slope(AF[t2], Y[t2, j], CL[t2]); c_all = np.polyfit(AF[trc], Y[trc, j], 1)[0]
        mafc = trc[M[trc, jaf]]; b_, a_ = np.polyfit(Y[mafc, jaf], AF[mafc], 1)   # AF ~ a + b * autofl label (training draw)
        m = M[G, j] & (ab[G] == mk); g = G[m]; y = Y[g, j]; p = P[m, j]; afh = a_ + b_ * P[m, jaf]; afh = afh - afh.mean()
        r0 = np.corrcoef(p, y)[0, 1]
        # what the TEST draw would want (oracle, for reference)
        tec = np.flatnonzero((ab == mk) & (bt == te_b) & M[:, j] & (hr < 0) & np.isfinite(AF)); t3 = tec[CL[tec] >= 0]; c_te_ly = within_slope(AF[t3], Y[t3, j], CL[t3])
        zaf = (P[m, jaf] - P[m, jaf].mean()) / P[m, jaf].std(); zp = (p - p.mean()) / p.std()
        sweep = {lam: np.corrcoef(zp + lam * zaf, y)[0, 1] for lam in (-.6, -.4, -.2, 0, .2, .4, .6)}
        lam_best = max(sweep, key=sweep.get)
        rows.append(dict(dir=d, target=mk, c_train_lymph=c_ly, c_train_all=c_all, c_test_lymph_ORACLE=c_te_ly, r_raw=r0,
                         r_debias_lymph=np.corrcoef(p - c_ly * afh, y)[0, 1], r_debias_all=np.corrcoef(p - c_all * afh, y)[0, 1],
                         oracle_best_lam=lam_best, oracle_best_r=sweep[lam_best], r_corr_pred_autofl=np.corrcoef(p, P[m, jaf])[0, 1]))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/c2_autofl_debias.csv', index=False); pd.set_option('display.width', 250)
print(df.round(3).to_string(index=False))
for c in ('r_raw', 'r_debias_lymph', 'r_debias_all', 'oracle_best_r'): print(c, df.groupby('dir')[c].sum().round(4).to_dict(), 'GOAL-delta(/22)', round((df[c] - df.r_raw).sum() / 22, 4))
