"""N036: re-score the GOAL-best predictions (N031 inductive base + zoom2 ensemble, and its members) against label versions that differ
only in the position detrend: v5 (GOAL as defined; in-sample split-0 HGB), v5c (same, Giemsa-free fits), v5cx (FOV-fold cross-fit,
Giemsa-free), v3c (no detrend; different per-slide scaling -> reference only). Same scorer as N031 score_inductive.py (common-11,
pooled Pearson over the held-out draw), plus the score restricted to test cells of split 0 vs splits 1+2 (the in-sample-detrended
cells vs never-fitted cells). Usage: n036_rescore.py OUTDIR"""
import sys, json, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'
VERS = ['targets_v5.npz', 'targets_v5c.npz', 'targets_v5cx.npz', 'targets_v3c.npz']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
rows = []
for v in VERS:
    T = np.load(W + '/' + v, allow_pickle=True); Y, M, sp = T['Y'], T['mask'], T['split']; names = [str(n) for n in T['names']]
    c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
    for d in ('b10', 'b01'):
        for arm, runs in [('base', [f'n031_base_{d}']), ('zoom2', [f'n031_zoom2_{d}']), ('ens', [f'n031_base_{d}', f'n031_zoom2_{d}'])]:
            Ps = [load(r) for r in runs]; G = Ps[0][1]; assert all((p[1] == G).all() for p in Ps); P = np.mean([p[0] for p in Ps], 0)
            for sub, keep in (('all', np.ones(len(G), bool)), ('split0', sp[G] == 0), ('split12', sp[G] > 0)):
                for j in c11:
                    m = M[G, j] & keep
                    rows.append(dict(targets=v.replace('targets_', '').replace('.npz', ''), dir=d, arm=arm, cells=sub, target=names[j], n=int(m.sum()),
                                     r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1]) if m.sum() > 50 else np.nan))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n036_rescore_per_target.csv', index=False)
g = df.groupby(['targets', 'arm', 'cells', 'dir']).r.mean().unstack('dir'); g['goal'] = g.mean(1)
g.round(4).to_csv(sys.argv[1] + '/n036_rescore_goal.csv'); pd.set_option('display.width', 200); print(g.round(4).to_string())
print(df[(df.arm == 'ens') & (df.cells == 'all')].pivot_table(index='target', columns='targets', values='r').round(3).to_string())
