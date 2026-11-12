"""N039 stage 2(i): clean in-distribution reference on the GOAL test cells. r50 SSL(n007, transductive -- reference only) FOV split,
v5 labels, full (318k) and data-matched (128k = b01's training size); scored per test draw on the same split-2 cells as the GOAL,
v5 common-11, pooled and within-FOV; compared with the N031 GOAL ensemble and N031 base."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, bt, fov = T['Y'], T['mask'], T['batch'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; rows = []
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
def wf(p, y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); return np.corrcoef(p - (np.bincount(inv, p) / c)[inv], y - (np.bincount(inv, y) / c)[inv])[0, 1]
arms = {'fov_v5_full': lambda d: load('n039_ceil_fov_v5_mt0'), 'fov_v5_128k': lambda d: load('n039_ceil_fov_v5_mt128000'),
        'goal_base': lambda d: load(f'n031_base_{d}'), 'goal_ens': lambda d: (np.mean([load(f'n031_{a}_{d}')[0] for a in ('base', 'zoom2')], 0), load(f'n031_base_{d}')[1])}
for d, b in (('b10', 0), ('b01', 1)):
    for arm, f in arms.items():
        P, G = f(d); k = bt[G] == b; P, G = P[k], G[k]
        for j in c11:
            m = M[G, j]; y = Y[G[m], j]; p = P[m, j]
            rows.append(dict(arm=arm, draw=b, target=names[j], r=float(np.corrcoef(p, y)[0, 1]), r_within=float(wf(p, y, fov[G[m]]))))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n039_ceiling_per_target.csv', index=False)
g = df.groupby(['arm', 'draw'])[['r', 'r_within']].mean().unstack('draw'); g[('r', 'mean')] = g['r'].mean(1); g[('r_within', 'mean')] = g['r_within'].mean(1)
pd.set_option('display.width', 200); print(g.round(4).to_string()); print(df.pivot_table(index='target', columns=['arm'], values='r').round(3).to_string())
