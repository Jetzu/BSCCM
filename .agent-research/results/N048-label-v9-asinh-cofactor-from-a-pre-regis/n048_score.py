"""N048: GOAL recipe trained on v9 (cofactor-150) labels vs N031 (v5-trained), scored on FROZEN v5 (the GOAL) and on v9; base, zoom2,
ensemble; paired Wilcoxon over 22 marker x direction items. Writes n048_scores.csv, n048_per_target.csv, plot CSV."""
import sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
W = '/hnvme/workspace/v181bb10-bsccm'; rows = []
for lab, tf in (('v5 (GOAL)', 'targets_v5.npz'), ('v9 cf150', 'targets_v5cx_cf150.npz')):
    T = np.load(f'{W}/{tf}', allow_pickle=True); Y, M = T['Y'], T['mask']; names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
    for model, pre in (('N031 (trained v5)', 'n031'), ('N048 (trained v9)', 'n048_v9')):
        for arm, mem in (('base', ['base']), ('zoom2', ['zoom2']), ('ens', ['base', 'zoom2'])):
            for d in ('b10', 'b01'):
                Ps = [np.load(f'{W}/runs/{pre}_{m}_{d}/eval_preds.npz') for m in mem]; G = Ps[0]['G']; P = np.mean([p['P'] for p in Ps], 0)
                for j in c11:
                    m = M[G, j]; rows.append(dict(labels=lab, model=model, arm=arm, dir=d, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n048_per_target.csv', index=False)
g = df.groupby(['labels', 'arm', 'model', 'dir']).r.mean().unstack('dir'); g['goal'] = g.mean(axis=1); g.round(4).to_csv(sys.argv[1] + '/n048_scores.csv')
pd.set_option('display.width', 200); print(g.round(4).to_string())
for lab in ('v5 (GOAL)', 'v9 cf150'):
    k = lambda mo: df[(df.labels == lab) & (df.model == mo) & (df.arm == 'ens')].set_index(['dir', 'target']).r.sort_index()
    x, y = k('N048 (trained v9)'), k('N031 (trained v5)'); print(f'{lab}: ens diff {(x - y).mean():+.4f}, better {(x > y).sum()}/22, Wilcoxon p {wilcoxon(x, y).pvalue:.3g}')
e = df[(df.labels == 'v5 (GOAL)') & (df.arm == 'ens')]; pd.DataFrame(dict(arm=e.model, value=e.r, seed=e.dir + ':' + e.target)).to_csv(sys.argv[1] + '/plot_n048_v5.csv', index=False)
print(df[(df.arm == 'ens')].pivot_table(index='target', columns=['labels', 'model'], values='r').round(3).to_string())
