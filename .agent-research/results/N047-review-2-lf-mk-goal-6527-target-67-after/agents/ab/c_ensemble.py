"""N047 AB (c): is the inductive ensemble saturated, or would a different member type decorrelate errors?
(1) residual correlation between members (per target x draw, residual = z(y) - fitted affine of z(p)); (2) add DIFFERENT inductive member
types already on disk to the 10-model ensemble (r18 supervised scratch v4 labels N013, r50 scratch v4 N014, low-pass input N042, 50 %/25 %
labels N040): weight chosen on the OTHER direction (no test-label selection); (3) output-shape: isotonic E[y|p] fitted on the other draw
(inductive) and 2-fold-FOV cross-fit on the test draw (oracle bound); rank-gauss transform; (4) what a decorrelated member could add (formula)."""
import sys, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm, rankdata
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, fov = T['Y'], T['mask'], T['fov']; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
dl = {'b10': 'b1to0', 'b01': 'b0to1'}
EXTRA = {'r18_sup_v4(N013,2 seeds)': lambda d: [f'n013_lf_sup_v4_{dl[d]}_s0', f'n013_lf_sup_v4_{dl[d]}_s1'], 'r50_scratch_v4(N014)': lambda d: [f'n014_r50_{dl[d]}'],
         'lowpass(N042)': lambda d: [f'n042_lp15_{d}'], 'lc50(N040)': lambda d: [f'n040_lc50_{d}'], 'lc25(N040)': lambda d: [f'n040_lc25_{d}']}
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return pd.DataFrame(E['P'], index=E['G'])
D = {}
for d in ('b10', 'b01'):
    mem = {m: load(f'{m}_{d}') for m in fam}
    for k, f in EXTRA.items(): L = [load(r) for r in f(d)]; mem[k] = sum(L) / len(L)
    common = sorted(set.intersection(*[set(v.index) for v in mem.values()])); D[d] = ({k: v.loc[common].values for k, v in mem.items()}, np.array(common))
    print(d, 'common cells', len(common), {k: len(v) for k, v in mem.items()})
z = lambda x: (x - x.mean()) / x.std()
def r_(p, y): return np.corrcoef(p, y)[0, 1]
rows = []; other = {'b10': 'b01', 'b01': 'b10'}; rng = np.random.default_rng(0)
for d in ('b10', 'b01'):
    mem, G = D[d]; Mo, Go = D[other[d]]
    for j in c11:
        m = M[G, j]; y = z(Y[G[m], j]); mo = M[Go, j]; yo = z(Y[Go[mo], j])
        P10 = np.stack([z(mem[k][m, j]) for k in fam]); ens = P10.mean(0)
        res = np.stack([y - np.polyval(np.polyfit(p, y, 1), p) for p in P10]); C = np.corrcoef(res); iu = np.triu_indices(10, 1)
        row = dict(dir=d, target=names[j], r_ens10=r_(ens, y), mean_member_r=np.mean([r_(p, y) for p in P10]), resid_corr_within10=C[iu].mean())
        ens_res = y - np.polyval(np.polyfit(ens, y, 1), ens)
        ens_o = np.mean([z(Mo[k][mo, j]) for k in fam], 0)
        for k in EXTRA:
            pe = z(mem[k][m, j]); pe_res = y - np.polyval(np.polyfit(pe, y, 1), pe)
            row[f'r_{k}'] = r_(pe, y); row[f'rescorr_{k}'] = r_(pe_res, ens_res)
            # weight chosen on the OTHER direction
            peo = z(Mo[k][mo, j]); ws = np.linspace(0, 0.5, 11)
            wbest = ws[np.argmax([r_((1 - w) * ens_o + w * peo, yo) for w in ws])]
            row[f'w_{k}'] = wbest; row[f'r_ens+{k}'] = r_((1 - wbest) * ens + wbest * pe, y); row[f'r_ens+{k}_w.2'] = r_(0.8 * ens + 0.2 * pe, y)
        # output shape: isotonic fitted on the other draw (inductive), cross-fit isotonic on test draw (oracle), rank-gauss
        iso = IsotonicRegression(out_of_bounds='clip').fit(ens_o, yo); row['r_iso_otherdraw'] = r_(iso.predict(ens), y)
        f = fov[G[m]]; u = np.unique(f); fold = dict(zip(u, rng.integers(0, 2, len(u)))); kf = np.array([fold[x] for x in f]); out = np.zeros(len(y))
        for q in (0, 1): out[kf == q] = IsotonicRegression(out_of_bounds='clip').fit(ens[kf != q], y[kf != q]).predict(ens[kf == q])
        row['r_iso_oracle_cf'] = r_(out, y); row['r_rankgauss'] = r_(norm.ppf((rankdata(ens) - .5) / len(ens)), y)
        rows.append(row)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/c_ensemble.csv', index=False); pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60)
print(df[['dir', 'target', 'r_ens10', 'mean_member_r', 'resid_corr_within10'] + [c for c in df.columns if c.startswith('rescorr_')]].round(3).to_string(index=False))
print(df[['dir', 'target', 'r_ens10'] + [c for c in df.columns if c.startswith('r_') and c not in ('r_ens10',) and 'rescorr' not in c]].round(3).to_string(index=False))
num = [c for c in df.columns if c.startswith('r_') or c.startswith('resid') or c.startswith('rescorr') or c.startswith('mean_')]
g = df.groupby('dir')[num].mean(); g.loc['GOAL'] = g.mean(); print(g.T.round(4).to_string())
