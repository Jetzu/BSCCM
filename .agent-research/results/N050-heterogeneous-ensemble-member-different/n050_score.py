"""N050: ConvNeXt-T member alone, its residual correlation with the ResNet-50 family, and its value when added (equal weight, fixed in
advance) to the N040 student ensemble and to the 10-model ensemble; frozen v5 GOAL, common-11."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
arms = {'N031 ensemble': fam[:2], 'N040 students': fam[4:6], 'ConvNeXt alone': ['n050_convnext'], 'N040 students + ConvNeXt': fam[4:6] + ['n050_convnext'],
        '10-model': fam, '10-model + ConvNeXt': fam + ['n050_convnext']}
rows = []; resid = []
for d in ('b10', 'b01'):
    L = {m: np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in fam + ['n050_convnext']}; G = L['n031_base']['G']; assert all(np.array_equal(v['G'], G) for v in L.values())
    for a, mem in arms.items():
        P = np.mean([L[m]['P'] for m in mem], 0)
        for j in c11:
            m = M[G, j]; rows.append(dict(arm=a, dir=d, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
    for j in c11:
        m = M[G, j]; y = Y[G[m], j]; e_c = L['n050_convnext']['P'][m, j] - y; e_f = np.mean([L[x]['P'][m, j] for x in fam], 0) - y
        resid.append(dict(dir=d, target=names[j], resid_r_convnext_vs_family=float(np.corrcoef(e_c, e_f)[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n050_per_target.csv', index=False); rd = pd.DataFrame(resid); rd.to_csv(sys.argv[1] + '/n050_residual_corr.csv', index=False)
g = df.groupby(['arm', 'dir']).r.mean().unstack(); g['goal'] = g.mean(axis=1); print(g.sort_values('goal').round(4).to_string())
print('residual r ConvNeXt vs 10-model mean:', rd.groupby('dir').resid_r_convnext_vs_family.mean().round(3).to_dict())
pd.DataFrame(dict(arm=df.arm, value=df.r, seed=df.dir + ':' + df.target))[df.arm.isin(['N040 students', 'N040 students + ConvNeXt'])].to_csv(sys.argv[1] + '/plot_n050.csv', index=False)
