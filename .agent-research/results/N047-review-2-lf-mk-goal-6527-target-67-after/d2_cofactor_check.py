"""N047 first-hand check of the assumption breaker's cofactor finding, independent code, NO position detrend: per single marker x test
draw, label = asinh(x / c) of the authors' unmixed value with c = split-0 q10 (as v3/v5) or c = 50; report FOV excess share and r of the
N031 GOAL ensemble prediction vs each label on the GOAL test cells. A cofactor effect independent of the detrend would show here."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
M, bt, fov, sp = T['mask'], T['batch'], T['fov'], T['split']; sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ab = pd.read_parquet(W + '/bsccm_index.parquet').antibodies.values.astype(str)
rng = np.random.default_rng(0)
def excess(y, f):
    u, inv, c = np.unique(f, return_inverse=True, return_counts=True); k = c[inv] >= 3; y, f = y[k], f[k]; u, inv = np.unique(f, return_inverse=True); n = np.bincount(inv)
    s = lambda v: np.var((np.bincount(inv, v) / n)[inv]) / np.var(v); return s(y) - np.mean([s(rng.permutation(y)) for _ in range(10)])
rows = []
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [np.load(f'{W}/runs/n031_{a}_{d}/eval_preds.npz') for a in ('base', 'zoom2')]; G = Ps[0]['G']; P = np.mean([p['P'] for p in Ps], 0)
    for m in ['CD3', 'CD56', 'CD19', 'CD45', 'CD14', 'CD16', 'HLA-DR', 'CD123']:
        j = names.index(f'single:{m}'); x = sm[f'{m}_single_antibody_model_unmixed'].values; s_all = (ab == m) & (bt == b) & M[:, j]
        q10 = np.percentile(x[s_all & (sp == 0)], 10); g = M[G, j]; gi = G[g]
        for cn, c in (('q10', q10), ('c50', 50.0)):
            y = np.arcsinh(x[gi] / c)
            rows.append(dict(dir=d, marker=m, cofactor=cn, c=round(float(c), 3), r=float(np.corrcoef(P[g, j], y)[0, 1]), fov_excess=float(excess(np.arcsinh(x[s_all] / c), fov[s_all]))))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/d2_cofactor_check.csv', index=False); pd.set_option('display.width', 200)
print(df.pivot_table(index=['marker'], columns=['dir', 'cofactor'], values=['r', 'fov_excess']).round(3).to_string())
print(df.groupby(['dir', 'cofactor']).r.mean().unstack().round(4))
