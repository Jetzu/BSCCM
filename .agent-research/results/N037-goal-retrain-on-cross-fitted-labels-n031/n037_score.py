"""N037: GOAL common-11 (N031 scorer: pooled Pearson over the held-out draw, mean of 11 targets, mean of b10/b01) for the N031 baseline
(v5-trained) and N037 (v5cx-trained), base / zoom2 / ensemble, scored on v5 (GOAL as defined), v5c and v5cx. Usage: n037_score.py OUTDIR"""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; rows = []
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
for v in ('v5', 'v5c', 'v5cx'):
    T = np.load(f'{W}/targets_{v}.npz', allow_pickle=True); Y, M = T['Y'], T['mask']; names = [str(n) for n in T['names']]
    c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
    for tag in ('n031', 'n037'):
        for d in ('b10', 'b01'):
            for arm, runs in (('base', [f'{tag}_base_{d}']), ('zoom2', [f'{tag}_zoom2_{d}']), ('ens', [f'{tag}_base_{d}', f'{tag}_zoom2_{d}'])):
                Ps = [load(r) for r in runs]; G = Ps[0][1]; assert all((p[1] == G).all() for p in Ps); P = np.mean([p[0] for p in Ps], 0)
                for j in c11:
                    m = M[G, j]; rows.append(dict(scored_on=v, model=tag, dir=d, arm=arm, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n037_per_target.csv', index=False)
g = df.groupby(['scored_on', 'arm', 'model', 'dir']).r.mean().unstack('dir'); g['goal'] = g.mean(1); g.round(4).to_csv(sys.argv[1] + '/n037_goal.csv')
pd.set_option('display.width', 200); print(g.round(4).to_string())
e = df[(df.arm == 'ens') & (df.scored_on == 'v5')].pivot_table(index='target', columns=['model', 'dir'], values='r'); print(e.round(3).to_string())
pd.concat([pd.DataFrame(dict(arm=f'{m} ({"v5 labels, N031" if m == "n031" else "v5cx labels, N037"})', value=df[(df.model == m) & (df.arm == 'ens') & (df.scored_on == 'v5')].groupby('dir').r.mean().values, seed=['b01', 'b10'])) for m in ('n031', 'n037')]).to_csv(sys.argv[1] + '/plot_goal_v5.csv', index=False)
