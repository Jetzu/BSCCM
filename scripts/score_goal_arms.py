"""Score GOAL arms on frozen v10 labels (targets_v5cx_v10.npz), common-11, mean over b10/b01, per seed; paired deltas vs a control arm.
usage: score_goal_arms.py OUTDIR control=run_s0,run_s1 armA=run_s0,run_s1 [armB=...]
  each run name is the prefix WITHOUT the _b10/_b01 suffix; seeds are paired by position.
writes OUTDIR/goal_arms.csv (arm, seed, direction, target, r), OUTDIR/goal_summary.csv, OUTDIR/plot_goal.csv (arm,value,seed)."""
import sys, os, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
out = sys.argv[1]; arms = dict(a.split('=', 1) for a in sys.argv[2:]); arms = {k: v.split(',') for k, v in arms.items()}
assert 'control' in arms, 'need control=...'
rows = []
for arm, runs in arms.items():
    for s, run in enumerate(runs):
        for d in ('b10', 'b01'):
            f = f'{W}/runs/{run}_{d}/eval_preds.npz'
            if not os.path.exists(f): print('MISSING', f); continue
            z = np.load(f); P, G = z['P'], z['G']
            for j in c11:
                m = M[G, j]; rows.append(dict(arm=arm, seed=s, run=run, direction=d, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(os.path.join(out, 'goal_arms.csv'), index=False)
g = df.groupby(['arm', 'seed', 'direction']).r.mean().unstack('direction'); g['GOAL'] = g.mean(axis=1); g = g.reset_index()
ok = g.dropna(subset=['GOAL'])
summ = ok.groupby('arm')[['b10', 'b01', 'GOAL']].mean(); summ['n_seeds'] = ok.groupby('arm').size()
c = ok[ok.arm == 'control'].set_index('seed').GOAL
summ['delta_vs_control_paired'] = [float((ok[ok.arm == a].set_index('seed').GOAL - c).dropna().mean()) for a in summ.index]
summ = summ.round(4); summ.to_csv(os.path.join(out, 'goal_summary.csv')); print(summ.to_string())
pt = df.groupby(['arm', 'target']).r.mean().unstack('arm'); print(pt.round(3).to_string())
ok.rename(columns={'GOAL': 'value'})[['arm', 'value', 'seed']].to_csv(os.path.join(out, 'plot_goal.csv'), index=False)
