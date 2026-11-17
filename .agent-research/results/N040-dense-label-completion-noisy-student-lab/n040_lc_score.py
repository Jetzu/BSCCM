"""N040 stage 1: label-count learning curve (N031 base recipe at 25/50/100 % of the training draw), frozen v5 GOAL scorer per target x
direction. Pre-registered kill: weak-target (CD19/CD56/CD123/HLA-DR) mean gain 50 % -> 100 % < .01."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; rows = []
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
for d in ('b10', 'b01'):
    for frac, run in ((25, f'n040_lc25_{d}'), (50, f'n040_lc50_{d}'), (100, f'n031_base_{d}')):
        P, G = load(run)
        for j in c11:
            m = M[G, j]; rows.append(dict(dir=d, frac=frac, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n040_lc_per_target.csv', index=False)
weak = ['single:CD19', 'single:CD56', 'single:CD123', 'single:HLA-DR']
df['group'] = np.where(df.target.isin(weak), 'weak4', 'other7')
g = df.groupby(['group', 'frac']).r.mean().unstack(); print(g.round(4)); goal = df.groupby(['frac', 'dir']).r.mean().unstack(); goal['goal'] = goal.mean(1); print(goal.round(4))
print(df.pivot_table(index='target', columns=['frac'], values='r').round(3).to_string())
gain = g.loc['weak4', 100] - g.loc['weak4', 50]; print('weak4 gain 50->100 %%: %.4f -> %s' % (gain, 'PASS (continue)' if gain >= .01 else 'KILL'))
pd.DataFrame(dict(arm=df.frac.astype(str) + ' %', value=df.r, seed=df.dir + ':' + df.target, x=df.frac)).to_csv(sys.argv[1] + '/plot_lc.csv', index=False)
