"""N059 ceilings (CPU): M1 greedy ensemble curve r_N over existing v10-scorable members + fit r_inf - c/N; M4 share of test-label
variance in the top 1/2/5 % of cells per target and direction. writes n059_m1.csv, n059_m4.csv"""
import numpy as np, pandas as pd, os
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fams = ['n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland', 'n053_cytoland_s1', 'n055_cyto_zoom2', 'n055_cyto_cf150', 'n051_v10_base',
        'n051_v10_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n031_base', 'n031_zoom2', 'n046_student2_base', 'n046_student2_zoom2',
        'n050_convnext', 'n053_dinobloom_b', 'n053_dinov3_b', 'n053_dinov3_s', 'n041_phys', 'n041_phystail', 'n037_base', 'n037_zoom2']
fams = [f for f in fams if all(os.path.exists(f'{W}/runs/{f}_{d}/eval_preds.npz') for d in ('b10', 'b01'))]
P = {d: {f: np.load(f'{W}/runs/{f}_{d}/eval_preds.npz')['P'] for f in fams} for d in ('b10', 'b01')}
G = {d: np.load(f'{W}/runs/{fams[0]}_{d}/eval_preds.npz')['G'] for d in ('b10', 'b01')}
def goal(sel):
    out = []
    for d in ('b10', 'b01'):
        Pm = np.mean([P[d][f] for f in sel], 0); g = G[d]
        out.append(np.mean([np.corrcoef(Pm[M[g, j], j], Y[g[M[g, j]], j])[0, 1] for j in c11]))
    return float(np.mean(out))
sel, rows = [], []
for k in range(min(10, len(fams))):
    best = max((f for f in fams if f not in sel), key=lambda f: goal(sel + [f])); sel.append(best); rows.append(dict(N=k + 1, added=best, goal=goal(sel)))
m1 = pd.DataFrame(rows); print(m1.round(4).to_string(index=False))
N = m1.N.values.astype(float); A = np.c_[np.ones_like(N), -1 / N]; coef = np.linalg.lstsq(A[1:], m1.goal.values[1:], rcond=None)[0]
print('fit r_inf = %.4f, c = %.4f (N>=2, greedy = optimistic, test-selected)' % tuple(coef)); m1['r_inf_fit'] = coef[0]; m1.to_csv('n059_m1.csv', index=False)
rows = []
for d in ('b10', 'b01'):
    g = G[d]
    for j in c11:
        y = Y[g[M[g, j]], j]; dev = (y - y.mean()) ** 2; o = np.sort(dev)[::-1]; n = len(y)
        rows.append(dict(direction=d, target=names[j], n=n, top1=o[:max(1, n // 100)].sum() / o.sum(), top2=o[:max(1, n // 50)].sum() / o.sum(),
                         top5=o[:max(1, n // 20)].sum() / o.sum(), n_top2=n // 50))
m4 = pd.DataFrame(rows); m4.to_csv('n059_m4.csv', index=False); print(m4.round(3).to_string(index=False))
