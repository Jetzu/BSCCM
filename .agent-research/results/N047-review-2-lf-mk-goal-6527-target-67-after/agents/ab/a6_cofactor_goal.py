import os; os.environ["OMP_NUM_THREADS"] = "4"
"""N047 AB (a6): full common-11 GOAL-equivalent when ONLY the asinh cofactor of the 8 single-antibody labels changes (q10 = v3/v5
reproduction, fixed 50, per-slide q50 from split-0 cells); grp targets stay v5. Same saved predictions, same cells. Models: N031 ensemble (GOAL
best), 10-model ensemble, n031_base single, n039 in-distribution 128k (draw gap). RE-BASE diagnostic, NOT GOAL progress."""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, M, bt, fov, sp = T['Y'], T['mask'], T['batch'], T['fov'], T['split']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
POS = ix[['fov_center_x', 'fov_center_y', 'fov_center_z', 'position_in_fov_x_pix', 'position_in_fov_y_pix']].values.astype(float)
c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return pd.DataFrame(E['P'], index=E['G'])
LAB = {}
for cn in ('q10', '50', 'q50'):
    Yc = Y5.copy()
    for m in ['CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3']:
        j = names.index(f'single:{m}'); x = sm[f'{m}_single_antibody_model_unmixed'].values
        for b in (0, 1):
            s = np.flatnonzero((ab == m) & (bt == b) & M[:, j]); tr = sp[s] == 0
            c = {'q10': np.percentile(x[s][tr], 10), '50': 50., 'q50': np.percentile(x[s][tr], 50)}[cn]
            t = np.arcsinh(x[s] / c); t = t - HGB(max_iter=100, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, random_state=0).fit(POS[s][tr], t[tr]).predict(POS[s])
            Yc[s, j] = (t - t[tr].mean()) / t[tr].std()
    LAB[cn] = Yc
rows = []
for d, b in (('b10', 0), ('b01', 1)):
    MOD = {'N031_ens': sum(load(f'{m}_{d}') for m in fam[:2]) / 2, 'HO10': sum(load(f'{m}_{d}') for m in fam) / 10, 'n031_base': load(f'n031_base_{d}')}
    IN = load('n039_ceil_fov_v5_mt128000'); MOD['IN128'] = IN.loc[IN.index.values[bt[IN.index.values] == b]]
    common = np.intersect1d(MOD['HO10'].index.values, MOD['IN128'].index.values)
    for k, P in MOD.items():
        for lab in ('v5', 'q10', '50', 'q50'):
            Yl = Y5 if lab == 'v5' else LAB[lab]; Gk = P.index.values
            for scope, G in (('own_cells', Gk), ('common_cells', common)):
                rs = [np.corrcoef(P.loc[G[M[G, j]]].values[:, j], Yl[G[M[G, j]], j])[0, 1] for j in c11]
                rows.append(dict(dir=d, model=k, labels=lab, scope=scope, r_mean=np.mean(rs), **{names[j]: r for j, r in zip(c11, rs)}))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/a6_cofactor_goal.csv', index=False); pd.set_option('display.width', 250)
g = df.pivot_table(index=['scope', 'model'], columns=['labels', 'dir'], values='r_mean')
for lab in ('v5', 'q10', '50', 'q50'): g[(lab, 'GOAL')] = (g[(lab, 'b10')] + g[(lab, 'b01')]) / 2
print(g.sort_index(axis=1).round(4).to_string())
