"""N047 AB (b): how much of the draw-shift gap (in-distribution N039 ceiling model vs inductive held-out ensemble, same test cells) is a
LOW-DIMENSIONAL label-definition difference rather than missing per-cell information?
Models on identical split-2 cells of each test draw: HO10 = 10-model inductive ensemble, HO31 = N031 best ensemble, IN128 = n039 FOV r50 v5
data-matched 128k (has seen both draws), INfull = n039 FOV 318k. Arms (all are ORACLE diagnostics that use test labels, 2-fold FOV cross-fit):
 raw | A: class-wise affine remap of the model's own output (3 LF classes from IN128's class head: 6 params) | A13: remap from ALL 13 heads of
 the same model (captures e.g. CD19-dose mixing in the pooled CD3/CD19/CD56) | B: labels residualised on the authors' autofluorescence
 (within class; oracle) | C: grp targets scored per slide (draw 1 has 2 'all' slides).
Gap = IN128 - HO10 per target, mean over common-11; share explained = 1 - gap_arm / gap_raw."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, fov, sl, bt = T['Y'], T['mask'], T['fov'], T['slide'], T['batch']; sn = T['slide_names'].astype(str)
c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); AF = np.arcsinh(sm['autofluor_single_antibody_model_unmixed'].values / 50); print('AF nan frac', np.isnan(AF).mean()); AF = np.nan_to_num(AF)
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G'], E['C3']
def ens(ms, d): L = [load(f'{m}_{d}') for m in ms]; return np.mean([l[0] for l in L], 0), L[0][1]
IN = {k: load(f'n039_ceil_fov_v5_{k}') for k in ('mt128000', 'mt0')}
rng = np.random.default_rng(0)
def cv_fit(X, y, f):
    u = np.unique(f); fold = dict(zip(u, rng.integers(0, 2, len(u)))); k = np.array([fold[x] for x in f]); out = np.zeros(len(y))
    X1 = np.c_[X, np.ones(len(y))]
    for q in (0, 1):
        tr = k != q; beta = np.linalg.lstsq(X1[tr], y[tr], rcond=None)[0]; out[~tr] = X1[~tr] @ beta
    return out
rows = []
for d, b in (('b10', 0), ('b01', 1)):
    MODELS = {'HO10': ens(fam, d), 'HO31': ens(fam[:2], d), 'IN128': IN['mt128000'][:2], 'INfull': IN['mt0'][:2]}
    common = np.intersect1d(MODELS['HO10'][1], IN['mt128000'][1]); common = common[bt[common] == b]
    at = {k: pd.Series(np.arange(len(G)), index=G)[common].values for k, (P, G) in MODELS.items()}
    C3 = IN['mt128000'][2][pd.Series(np.arange(len(IN['mt128000'][1])), index=IN['mt128000'][1])[common].values]
    oh = np.eye(3)[C3]
    for j in c11:
        m = M[common, j]; g = common[m]; y = Y[g, j]; f = fov[g]; c = oh[m]
        afc = c * (AF[g][:, None] - np.array([AF[g][c[:, q] > 0].mean() if (c[:, q] > 0).any() else 0 for q in range(3)])[None, :])   # autofl centred within class
        X1 = np.c_[c, afc]; beta = np.linalg.lstsq(X1, y, rcond=None)[0]; yB = y - afc @ beta[3:]   # remove ONLY the within-class autofl slope (oracle, full fit)
        for k, (P, G) in MODELS.items():
            Pk = P[at[k]][m]; p = Pk[:, j]
            r = dict(draw=b, target=names[j], model=k, n=len(g), raw=np.corrcoef(p, y)[0, 1])
            r['A_class_remap'] = np.corrcoef(cv_fit(np.c_[c[:, :2], c * p[:, None]], y, f), y)[0, 1]
            r['A13_allheads'] = np.corrcoef(cv_fit(Pk, y, f), y)[0, 1]
            r['B_autofl_resid'] = np.corrcoef(p, yB)[0, 1]
            if names[j].startswith('grp:') and len(np.unique(sl[g])) > 1:
                for s in np.unique(sl[g]): ms = sl[g] == s; r[f'C_slide_{sn[s].split("/")[-1]}'] = np.corrcoef(p[ms], y[ms])[0, 1]
            rows.append(r)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/b_draw_gap.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
arms = ['raw', 'A_class_remap', 'A13_allheads', 'B_autofl_resid']
piv = df.pivot_table(index=['draw', 'target'], columns='model', values=arms)
for b in (0, 1):
    print(f'=== test draw {b} ({"b10" if b == 0 else "b01"}) per-target gap IN128-HO10 and INfull-HO10 ===')
    t = pd.DataFrame({a: piv.loc[b][(a, 'IN128')] - piv.loc[b][(a, 'HO10')] for a in arms}); t['INfull-HO10 raw'] = piv.loc[b][('raw', 'INfull')] - piv.loc[b][('raw', 'HO10')]
    t['HO10 raw'] = piv.loc[b][('raw', 'HO10')]; t['IN128 raw'] = piv.loc[b][('raw', 'IN128')]
    print(t.round(3).to_string()); print('MEAN', t.mean().round(4).to_dict())
    for a in arms: print(f'  mean over 11: {a:16s}', {k: round(piv.loc[b][(a, k)].mean(), 4) for k in ['HO10', 'HO31', 'IN128', 'INfull']})
for b in (0, 1):
    q = piv.loc[b]; print(f'draw {b} HO10/IN128 per target raw -> A13:'); print(pd.DataFrame({'HO10_raw': q[('raw','HO10')], 'HO10_A13': q[('A13_allheads','HO10')], 'IN128_raw': q[('raw','IN128')], 'IN128_A13': q[('A13_allheads','IN128')], 'HO10_B': q[('B_autofl_resid','HO10')], 'IN128_B': q[('B_autofl_resid','IN128')]}).round(3).to_string())
gp = df[df.target.str.startswith('grp:') & (df.draw == 1)]; print(gp[['target', 'model', 'raw'] + [c for c in df.columns if c.startswith('C_')]].round(3).to_string(index=False))
