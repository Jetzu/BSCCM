"""N038 D2 (independent reproduction of assumption-breaker claims b/c): for the GOAL ensemble (N031 base+zoom2) on its test cells, v5
labels, common-11: (1) pooled r (= GOAL); (2) within-FOV r (label and prediction demeaned per FOV, FOVs with >= 3 labelled cells);
(3) FOV-mean variance share of the LABEL and of the PREDICTION, each minus a permutation null (FOV ids shuffled within the target's
cells, 20 perms). If label FOV share >> prediction FOV share and within-FOV r >> pooled r, FOV-level label offsets cap the GOAL."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, fov = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; rng = np.random.default_rng(0)
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
def share(v, f):
    u, inv = np.unique(f, return_inverse=True); mu = np.bincount(inv, v) / np.bincount(inv); return np.var(mu[inv]) / np.var(v)
def demean(v, f):
    u, inv = np.unique(f, return_inverse=True); return v - (np.bincount(inv, v) / np.bincount(inv))[inv]
rows = []
for d in ('b10', 'b01'):
    Ps = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; G = Ps[0][1]; P = np.mean([p[0] for p in Ps], 0)
    for j in c11:
        m = M[G, j]; g = G[m]; y = Y[g, j].astype(float); p = P[m, j].astype(float); f = fov[g]
        u, inv, c = np.unique(f, return_inverse=True, return_counts=True); k = c[inv] >= 3
        null_y = np.mean([share(y[k], rng.permutation(f[k])) for _ in range(20)]); null_p = np.mean([share(p[k], rng.permutation(f[k])) for _ in range(20)])
        rows.append(dict(dir=d, target=names[j], n=int(m.sum()), r_pooled=np.corrcoef(p, y)[0, 1],
                         r_within=np.corrcoef(demean(p[k], f[k]), demean(y[k], f[k]))[0, 1],
                         label_fov_excess=share(y[k], f[k]) - null_y, pred_fov_excess=share(p[k], f[k]) - null_p))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/d2_within_fov.csv', index=False); pd.set_option('display.width', 200)
print(df.round(3).to_string(index=False)); g = df.groupby('dir')[['r_pooled', 'r_within']].mean(); print(g.round(4)); print('GOAL pooled', round(g.r_pooled.mean(), 4), 'within-FOV', round(g.r_within.mean(), 4))
