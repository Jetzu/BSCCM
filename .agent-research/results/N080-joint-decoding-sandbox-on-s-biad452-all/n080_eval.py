"""N080 post-hoc evaluation (red-team fixes): joint decoding on held-out donors of S-BIAD452, beyond cell type.
For each fold (test donor d, training donor t = (d+2) % 3, as in train_ifc_sandbox.donor_split):
- Targets B: asinh(I_raw / c_t) standardised with the TRAINING donor's mean/sd (all its cells; evaluation-side definition, identical for
  every arm). Targets A (the run's own per-run z, Y in eval_preds) reported alongside for per-marker r.
- Valley thresholds per marker: 2-component Gaussian mixture on the training donor's B values; threshold = density minimum between the
  two component means (fallback: posterior 0.5 crossing). True population fractions reported per fold.
- Fine types (TRUE labels, evaluation only): eos Siglec8+; neut CD15+ Siglec8-; mono CD14+ CD15-; B CD19+ CD3- CD14- CD15-;
  T4 CD3+ CD4+ CD8-; T8 CD3+ CD8+ CD4-; Tother CD3+ else; NK CD56+ CD3- CD19- CD14- CD15-; other.
- TYPE-ORACLE baseline: each test cell gets the training donor's mean B profile of its TRUE fine type (upper bound of what cell-type
  knowledge alone gives).
- Model (and kNN) metrics on B: per-marker r, macro r, model - type-oracle per marker; WITHIN-fine-type r per marker (types with
  >= 100 test cells; the oracle has zero within-type variance) = decoding beyond cell type; within-type pair correlations predicted vs
  true for (CD4,CD8) in T cells and (CD3,CD19), (CD3,CD56), (CD19,CD56) in lymphocytes (T4+T8+Tother+B+NK).
usage: n080_eval.py RUNS_DIR PACK_DIR OUT_DIR   (runs named n080_<regime>_d<d>)
writes n080_eval_marker.csv, n080_eval_withintype.csv, n080_eval_pairs.csv, n080_eval_summary.csv, n080_eval_populations.csv"""
import sys, os, re, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
RUNS, PACK, OUT = sys.argv[1:4]
M = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']; ix = {m: i for i, m in enumerate(M)}
lab = np.load(f'{PACK}/labels.npz', allow_pickle=True); I = lab['I_raw'].astype(np.float64); samp = lab['sample'].astype(str)
donor = np.array([int(re.match(r'sample(\d+)', s).group(1)) for s in samp])
def pr(a, b): a, b = a - a.mean(), b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 1e-12 else np.nan
def valley(x):
    g = GaussianMixture(2, random_state=0).fit(x[:, None]); mu = np.sort(g.means_.ravel())
    grid = np.linspace(mu[0], mu[1], 400); dens = np.exp(g.score_samples(grid[:, None]))
    return float(grid[np.argmin(dens)]) if mu[1] - mu[0] > 1e-6 else float(np.median(x))
def types(Yb, thr):
    pos = {m: Yb[:, ix[m]] > thr[m] for m in M}; n = len(Yb); t = np.array(['other'] * n, dtype=object)
    t[pos['CD3'] & ~(pos['CD4'] & ~pos['CD8']) & ~(pos['CD8'] & ~pos['CD4'])] = 'Tother'
    t[pos['CD3'] & pos['CD8'] & ~pos['CD4']] = 'T8'; t[pos['CD3'] & pos['CD4'] & ~pos['CD8']] = 'T4'
    t[pos['CD56'] & ~pos['CD3'] & ~pos['CD19'] & ~pos['CD14'] & ~pos['CD15']] = 'NK'
    t[pos['CD19'] & ~pos['CD3'] & ~pos['CD14'] & ~pos['CD15']] = 'B'
    t[pos['CD14'] & ~pos['CD15']] = 'mono'; t[pos['CD15'] & ~pos['Siglec8']] = 'neut'; t[pos['Siglec8']] = 'eos'
    return t
LYM = ('T4', 'T8', 'Tother', 'B', 'NK'); PAIRS = [('T', 'CD4', 'CD8'), ('lymph', 'CD3', 'CD19'), ('lymph', 'CD3', 'CD56'), ('lymph', 'CD19', 'CD56')]
rows_m, rows_w, rows_p, rows_s, rows_pop = [], [], [], [], []
for d in (0, 1, 2):
    t_d = (d + 2) % 3; trm = donor == t_d
    c = np.maximum(1.4826 * np.median(np.abs(I[trm] - np.median(I[trm], 0)), 0), 1.0); Z = np.arcsinh(I / c)
    mu, sd = Z[trm].mean(0), Z[trm].std(0) + 1e-6; Yb_all = (Z - mu) / sd
    thr = {m: valley(Yb_all[trm, ix[m]]) for m in M}
    ttr = types(Yb_all[trm], thr); prof = {k: Yb_all[trm][ttr == k].mean(0) for k in np.unique(ttr)}
    for regime in ('full', 'single', 'single_pooled'):
        f = f'{RUNS}/n080_{regime}_d{d}/eval_preds.npz'
        if not os.path.exists(f): print('MISSING', f); continue
        z = np.load(f, allow_pickle=True); rows = z['rows']; Yb = Yb_all[rows]; Ya = z['Y']; tt = types(Yb, thr)
        if regime == 'full':
            for k in np.unique(tt): rows_pop.append(dict(test_donor=d, type=k, frac_test=float((tt == k).mean()), frac_train=float((ttr == k).mean())))
            for m in M: rows_pop.append(dict(test_donor=d, type='thr:' + m, frac_test=float((Yb[:, ix[m]] > thr[m]).mean()), frac_train=float((Yb_all[trm, ix[m]] > thr[m]).mean())))
        oracle = np.stack([prof.get(k, np.zeros(9)) for k in tt])
        preds = {'model': z['P']}
        if 'P_knn' in z.files: preds['knn'] = z['P_knn']
        for name, P in preds.items():
            arm = f'{regime}:{name}'
            for m in M:
                j = ix[m]; rows_m.append(dict(test_donor=d, arm=arm, marker=m, r_B=pr(P[:, j], Yb[:, j]), r_A=pr(P[:, j], Ya[:, j]), r_type_oracle=pr(oracle[:, j], Yb[:, j])))
                for k in np.unique(tt):
                    s = tt == k
                    if s.sum() >= 100 and Yb[s, j].std() > 1e-6: rows_w.append(dict(test_donor=d, arm=arm, type=k, marker=m, n=int(s.sum()), r_within=pr(P[s, j], Yb[s, j])))
            for grp, a, b in PAIRS:
                s = np.isin(tt, ('T4', 'T8', 'Tother')) if grp == 'T' else np.isin(tt, LYM)
                if s.sum() >= 100: rows_p.append(dict(test_donor=d, arm=arm, group=grp, pair=f'{a}-{b}', n=int(s.sum()), r_true=pr(Yb[s, ix[a]], Yb[s, ix[b]]), r_pred=pr(P[s, ix[a]], P[s, ix[b]])))
dm = pd.DataFrame(rows_m); dw = pd.DataFrame(rows_w); dp = pd.DataFrame(rows_p); dpop = pd.DataFrame(rows_pop)
os.makedirs(OUT, exist_ok=True)
for df, n in ((dm, 'marker'), (dw, 'withintype'), (dp, 'pairs'), (dpop, 'populations')): df.to_csv(f'{OUT}/n080_eval_{n}.csv', index=False)
summ = dm.groupby('arm').agg(macro_r_B=('r_B', 'mean'), macro_r_A=('r_A', 'mean'), type_oracle=('r_type_oracle', 'mean'))
summ['model_minus_oracle'] = summ.macro_r_B - summ.type_oracle
summ['within_type_r'] = dw.groupby('arm').r_within.mean()
lym = dw[dw.type.isin(LYM)]; summ['within_lymph_types_r'] = lym.groupby('arm').r_within.mean()
dp['pair_err'] = (dp.r_pred - dp.r_true).abs(); summ['pair_corr_abs_err'] = dp.groupby('arm').pair_err.mean()
summ = summ.round(4); summ.to_csv(f'{OUT}/n080_eval_summary.csv'); pd.set_option('display.width', 200); print(summ.to_string())
print(dpop[dpop.type.str.startswith('thr:') == False].pivot_table(index='type', columns='test_donor', values='frac_test').round(3).to_string())
