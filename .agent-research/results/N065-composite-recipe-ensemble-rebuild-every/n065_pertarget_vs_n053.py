"""N065 descriptive: per-target E3c and 6-way vs the N053 GOAL-best ensemble (n048_v9_base + n048_v9_zoom2 + n053_cytoland).
writes n065_vs_n053_per_target.csv, plot_n065_goal.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
rows = []
for d in ('b10', 'b01'):
    P = np.mean([np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')], 0)
    G = np.load(f'{W}/runs/n053_cytoland_{d}/eval_preds.npz')['G']
    for j in c11: rows.append(dict(set='N053 ens (best)', direction=d, target=names[j], r=float(np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1])))
pt = pd.concat([pd.read_csv('n065_per_target.csv'), pd.DataFrame(rows)])
w = pt.groupby(['set', 'target']).r.mean().unstack(0)
w['E3c-N053'] = w['E3c (PRIMARY)'] - w['N053 ens (best)']; w['6way-N053'] = w['E3c + N053 (6-way)'] - w['N053 ens (best)']
print(w[['N053 ens (best)', 'E3c (PRIMARY)', 'E3c + N053 (6-way)', 'E3c-N053', '6way-N053']].round(4).to_string()); w.round(4).to_csv('n065_vs_n053_per_target.csv')
g = pt.groupby(['set', 'direction']).r.mean().reset_index()
print(g.groupby('set').r.mean().round(4).to_string())
g.rename(columns=dict(set='arm', r='value', direction='seed')).to_csv('plot_n065_goal.csv', index=False)
