"""N047 AB (b2): robust version of the readout-remap decomposition + few-shot curve + CD19-dose check.
A13r: per target, ridge remap of ALL 13 output heads of a model (z-scored), fitted on test-draw labels, 2-fold FOV cross-fit, averaged
over 10 fold seeds (ORACLE diagnostic). Few-shot: fit the same remap on k labelled cells drawn from a random half of the test FOVs, score
on the other half (20 repeats), k = 25..1000 per target -- how many labelled cells of a NEW isolation buy the draw gap.
Dose: grp:CD3/CD19/CD56 in batch-1 'all' slides used CD19 at 35 % dose -> r(p_grp - lam * z(p_CD19), y) for lam sweep."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, fov, sl, bt = T['Y'], T['mask'], T['fov'], T['slide'], T['batch']
c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return pd.DataFrame(E['P'], index=E['G'])
z = lambda x: (x - x.mean(0)) / x.std(0)
def remap_r(X, y, tr, te, own):
    Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-9); mdl = Ridge(alpha=max(1.0, 0.05 * tr.sum())).fit(Xs[tr], y[tr]); return np.corrcoef(mdl.predict(Xs[te]), y[te])[0, 1]
rows, fs = [], []
for d, b in (('b10', 0), ('b01', 1)):
    HO = sum(load(f'{m}_{d}') for m in fam) / len(fam); IN = load('n039_ceil_fov_v5_mt128000')
    common = np.intersect1d(HO.index.values, IN.index.values); common = common[bt[common] == b]
    MOD = {'HO10': HO.loc[common].values, 'IN128': IN.loc[common].values}
    for j in c11:
        m = M[common, j]; g = common[m]; y = Y[g, j]; f = fov[g]; u = np.unique(f)
        for k, P in MOD.items():
            X = P[m]; raw = np.corrcoef(X[:, j], y)[0, 1]; a13 = []
            for s in range(10):
                rs = np.random.default_rng(s); h = dict(zip(u, rs.integers(0, 2, len(u)))); kk = np.array([h[x] for x in f])
                rr = []
                for q in (0, 1):
                    te = kk == q; rr.append((remap_r(X, y, ~te, te, j), te.sum()))
                # score = r on the concatenated? use mean of the two halves' r and the raw r on the same halves
                a13.append(np.mean([x[0] for x in rr]) - np.mean([np.corrcoef(X[kk == q, j], y[kk == q])[0, 1] for q in (0, 1)]))
            rows.append(dict(draw=b, target=names[j], model=k, n=len(y), raw=raw, A13_gain=np.mean(a13), A13_gain_sd=np.std(a13)))
            if k == 'HO10':
                for kshot in (25, 50, 100, 250, 500, 1000):
                    gains = []
                    for s in range(20):
                        rs = np.random.default_rng(100 + s); h = dict(zip(u, rs.integers(0, 2, len(u)))); kk = np.array([h[x] for x in f])
                        pool = np.flatnonzero(kk == 0); te = kk == 1
                        if len(pool) < kshot: continue
                        trm = np.zeros(len(y), bool); trm[rs.choice(pool, kshot, replace=False)] = True
                        gains.append(remap_r(X, y, trm, te, j) - np.corrcoef(X[te, j], y[te])[0, 1])
                    if gains: fs.append(dict(draw=b, target=names[j], k=kshot, gain=np.mean(gains), sd=np.std(gains), n_rep=len(gains)))
        if names[j] == 'grp:CD3/CD19/CD56' and b == 1:
            j19 = names.index('single:CD19')
            for k, P in MOD.items():
                X = P[m]
                for s_ in np.unique(sl[g]):
                    ms = sl[g] == s_
                    out = {lam: np.corrcoef(z(X[ms, j]) - lam * z(X[ms, j19]), y[ms])[0, 1] for lam in (0, .1, .2, .3, .5, .65)}
                    print('dose check', k, T['slide_names'][s_], {l: round(v, 4) for l, v in out.items()})
df = pd.DataFrame(rows); df['remapped'] = df.raw + df.A13_gain; df.to_csv(sys.argv[1] + '/b2_readout.csv', index=False)
fsd = pd.DataFrame(fs); fsd.to_csv(sys.argv[1] + '/b2_fewshot.csv', index=False); pd.set_option('display.width', 250)
for b in (0, 1):
    q = df[df.draw == b].pivot(index='target', columns='model', values=['raw', 'remapped', 'A13_gain_sd'])
    q['gap_raw'] = q[('raw', 'IN128')] - q[('raw', 'HO10')]; q['gap_remapped'] = q[('remapped', 'IN128')] - q[('remapped', 'HO10')]
    print(f'== draw {b}'); print(q.round(3).to_string()); print('MEAN', q.mean().round(4).to_dict())
    ex = q.drop(index='single:CD123'); print('MEAN excl CD123', ex[['gap_raw', 'gap_remapped']].mean().round(4).to_dict())
print(fsd.groupby(['draw', 'k']).gain.mean().unstack('draw').round(4).to_string())
print(fsd[fsd.target != 'single:CD123'].groupby(['draw', 'k']).gain.mean().unstack('draw').round(4).to_string())
