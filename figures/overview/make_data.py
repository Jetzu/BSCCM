"""Overview figure data (BSCCM, 2026-11-24): writes figures/overview/data/*.csv from the notebook log and frozen eval predictions.
progress.csv: every measured GOAL attempt (item, value, label era, single model vs ensemble).
per_target.csv: per-target Pearson r (v10, common-11) of the headline models per direction.
mm3.csv: multi-marker metric (3 pooled panels) per model and seed."""
import json, numpy as np, pandas as pd, os
R = '/home/hpc/v181bb/v181bb10/BSCCM'; OUT = R + '/figures/overview/data'; W = '/hnvme/workspace/v181bb10-bsccm'
# Hand-checked list (N086, red-team rounds 1-2): per item, its BEST single-network arm (seed mean where several seeds) and its own
# ensembles / composites, each on the label version the item scored it on, from the item's notebook text / result CSVs.
# kind: single model = one network; multi-network = averaged ensemble or composite (N063: pooled groups from one network, singles
# from another). Omitted: v4 era (N007-N022: 12-marker metric on v4 labels; N010/N015/N020/N021 carry the standing best) and the N082
# half-label diagnostics. Multi-network = each item's PRIMARY combination only (secondary/descriptive ones listed in the fig1
# caption). Single model: per-direction encoder choices only where pre-registered (N084), not N083's post-hoc .6753.
# N048 was measured on frozen v5 (.6522 ens / .6456 base); its v10 number .6804 is N051's re-score, plotted at
# N051 with N051's own v10 retrain (.6788 ens, base seed 0 .6730).
C = [('N023', .6404, 'single model'), ('N025', .6390, 'single model'),
     ('N026', .6422, 'single model'), ('N026', .6515, 'multi-network'), ('N031', .6442, 'single model'), ('N031', .6527, 'multi-network'),
     ('N037', .6445, 'single model'), ('N037', .6537, 'multi-network'), ('N040', .6529, 'single model'), ('N040', .6564, 'multi-network'),
     ('N041', .6450, 'single model'), ('N041', .6531, 'multi-network'), ('N042', .6411, 'single model'),
     ('N046', .6510, 'single model'), ('N046', .6546, 'multi-network'), ('N047', .6571, 'multi-network'),
     ('N048', .6456, 'single model'), ('N048', .6522, 'multi-network'), ('N050', .6135, 'single model'), ('N050', .6533, 'multi-network'),
     ('N051', .6730, 'single model'), ('N051', .6804, 'multi-network'), ('N051', .6788, 'multi-network'),
     ('N053', .6660, 'single model'), ('N053', .6862, 'multi-network'), ('N055', .6704, 'single model'), ('N055', .6902, 'multi-network'),
     ('N060', .6756, 'single model'), ('N061', .6742, 'single model'), ('N062', .6704, 'single model'),
     ('N063', .6766, 'single model'), ('N063', .6784, 'multi-network'),
     ('N065', .6899, 'multi-network'), ('N067', .6624, 'single model'),
     ('N068', .6718, 'single model'), ('N073', .6658, 'single model'), ('N074', .6683, 'single model'), ('N075', .6708, 'single model'),
     ('N076', .6681, 'single model'), ('N078', .6707, 'single model'), ('N081', .6673, 'single model'), ('N083', .6681, 'single model'),
     ('N084', .6744, 'single model'), ('N084', .6911, 'multi-network'), ('N085', .6751, 'single model'), ('N085', .6967, 'multi-network')]
prog = []
for k, v, kind in C:
    n = int(k[1:]); era = 'v10 labels' if n >= 51 else 'v5 labels'
    prog.append(dict(item=k, n=n, value=v, era=era, kind=kind))
pd.DataFrame(prog).to_csv(OUT + '/progress.csv', index=False)
T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
grp = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
# each model = list of members; a member = runs whose predictions are averaged (ensemble); r is averaged over members (seeds)
ENS3 = ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')
MODELS = {'ResNet-50 (seeds 0, 1)': {d: [[f'n051_v10_base_{d}'], [f'n051_v10_base_s1_{d}']] for d in ('b10', 'b01')},
          'ConvNeXt-S (seeds 0, 1)': {d: [[f'n078_ft_cnxs_60_s0_{d}'], [f'n083_cnxs60_s1_{d}']] for d in ('b10', 'b01')},
          'Ensemble N053': {d: [[f'{r}_{d}' for r in ENS3]] for d in ('b10', 'b01')},
          'Ensemble N084': {d: [[f'{r}_{d}' for r in ENS3] + [('n083_cnxs60_s1_b10' if d == 'b10' else 'n084_cnxs120_s1_b01')]] for d in ('b10', 'b01')}}
pt = []
for mname, spec in MODELS.items():
    for d, members in spec.items():
        rr = []
        for runs in members:
            Z = [L(r) for r in runs]; G = Z[0]['G']; assert all((z['G'] == G).all() for z in Z); P = np.mean([z['P'] for z in Z], 0)
            rr.append([float(np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1]) for j in c11])
        for jj, j in enumerate(c11):
            pt.append(dict(model=mname, direction=d, target=names[j].replace('single:', '').replace('grp:', 'pool '), r=float(np.mean([x[jj] for x in rr]))))
pd.DataFrame(pt).to_csv(OUT + '/per_target.csv', index=False)
ga = pd.read_csv(R + '/.agent-research/results/N083-convnext-s-in-domain-ssl-encoder-for-mul/goal_arms.csv')
mm = ga[ga.target.isin(['grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16'])].groupby(['arm', 'seed']).r.mean().reset_index()
mm['arm'] = mm.arm.map({'control': 'ResNet-50 (control)', 'cnxs60': 'ConvNeXt-S'}); mm.rename(columns={'r': 'MM3'}).to_csv(OUT + '/mm3.csv', index=False)
print(pd.DataFrame(prog).groupby(['era', 'kind']).size()); print(pd.DataFrame(pt).groupby('model').r.mean().round(4)); print(mm)
