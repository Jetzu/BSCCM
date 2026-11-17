"""N047 diagnosis: GOAL of a large ensemble of every inductive held-out model on the frozen v5 GOAL (pooled and within-FOV), to test
whether the predictor family is saturated against these labels. Members: N031 base/zoom2, N037 base/zoom2, N040 students, N041 phys,
N046 students (all inductive SSL init, same test cells)."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, fov = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
def wf(p, y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); return np.corrcoef(p - (np.bincount(inv, p) / c)[inv], y - (np.bincount(inv, y) / c)[inv])[0, 1]
rows = []
for name, members in [('N031 ensemble (best)', fam[:2]), ('all 10 models', fam), ('N040 students', fam[4:6])]:
    for d in ('b10', 'b01'):
        Ps = [np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in members]; G = Ps[0]['G']; P = np.mean([p['P'] for p in Ps], 0)
        for j in c11:
            m = M[G, j]; rows.append(dict(ens=name, dir=d, target=names[j], r=np.corrcoef(P[m, j], Y[G[m], j])[0, 1], r_within=wf(P[m, j], Y[G[m], j], fov[G[m]])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/d1_big_ensemble.csv', index=False)
g = df.groupby(['ens', 'dir'])[['r', 'r_within']].mean().unstack('dir'); g[('r', 'goal')] = g['r'].mean(axis=1); g[('r_within', 'goal')] = g['r_within'].mean(axis=1)
pd.set_option('display.width', 200); print(g.round(4).to_string())
