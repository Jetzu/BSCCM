"""N040 + N041 scoring on the frozen GOAL (v5, common-11, pooled Pearson over the held-out draw, mean of b10/b01).
N041: physics aug + consistency (phys) and + tail05 (phystail), base model, vs N031 base; also each blended with N031 zoom2 (ensemble analog).
N040: noisy student (pseudo labels from the N031 ensemble on the training draw + phys aug), base + zoom2, vs N031 base / zoom2 / ensemble.
Writes <out>/scores_per_target.csv, <out>/scores_goal.csv and plot CSVs (per target x direction, paired)."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
arms = {'N031 base': ['n031_base_{d}'], 'N031 ensemble': ['n031_base_{d}', 'n031_zoom2_{d}'],
        'N041 phys': ['n041_phys_{d}'], 'N041 phys+tail': ['n041_phystail_{d}'],
        'N041 phys + N031 zoom2': ['n041_phys_{d}', 'n031_zoom2_{d}'], 'N041 phys+tail + N031 zoom2': ['n041_phystail_{d}', 'n031_zoom2_{d}'],
        'N040 student base': ['n040_student_base_{d}'], 'N040 student zoom2': ['n040_student_zoom2_{d}'], 'N031 zoom2': ['n031_zoom2_{d}'],
        'N040 student ensemble': ['n040_student_base_{d}', 'n040_student_zoom2_{d}']}
rows = []
for arm, runs in arms.items():
    for d in ('b10', 'b01'):
        Ps = [load(r.format(d=d)) for r in runs]; G = Ps[0][1]; assert all(np.array_equal(p[1], G) for p in Ps); P = np.mean([p[0] for p in Ps], 0)
        for j in c11:
            m = M[G, j]; rows.append(dict(arm=arm, dir=d, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/scores_per_target.csv', index=False)
g = df.groupby(['arm', 'dir']).r.mean().unstack(); g['goal'] = g.mean(axis=1); g = g.sort_values('goal'); g.round(4).to_csv(sys.argv[1] + '/scores_goal.csv')
pd.set_option('display.width', 200); print(g.round(4).to_string())
weak = ['single:CD19', 'single:CD56', 'single:CD123', 'single:HLA-DR']
print(df[df.target.isin(weak)].pivot_table(index='target', columns='arm', values='r').round(3).T.to_string())
for out, sel in ((sys.argv[1] + '/plot_n041.csv', ['N031 base', 'N041 phys', 'N041 phys+tail']), (sys.argv[2] + '/plot_n040.csv', ['N031 ensemble', 'N040 student ensemble'])):
    s = df[df.arm.isin(sel)]; pd.DataFrame(dict(arm=s.arm, value=s.r, seed=s.dir + ':' + s.target)).to_csv(out, index=False)
