"""N053: residual correlation of each new encoder with the r50 family (N048 base) and FOV-bootstrap of GOAL differences
(N048 + X vs N048 + best r50-family third member n040_student_base). usage: n053_decorr.py OUTCSV"""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, FOV = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
L = lambda r, d: np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')
rows = []; rng = np.random.default_rng(0)
def goal_parts(P, G):
    return {j: (P[M[G, j], j], Y[G[M[G, j]], j], FOV[G[M[G, j]]]) for j in c11}
def r_(p, y): return np.corrcoef(p, y)[0, 1]
for X in ['n053_cytoland', 'n053_dinobloom_b', 'n053_dinov3_b', 'n053_dinov3_s', 'n050_convnext', 'n040_student_base', 'n051_v10_base']:
    for d in ('b10', 'b01'):
        a, x = L('n048_v9_base', d), L(X, d); G = a['G']; assert np.array_equal(G, x['G'])
        res = [np.corrcoef(a['P'][M[G, j], j] - Y[G[M[G, j]], j], x['P'][M[G, j], j] - Y[G[M[G, j]], j])[0, 1] for j in c11]
        rows.append(dict(member=X, dir=d, resid_r_vs_n048base=float(np.mean(res))))
df = pd.DataFrame(rows).groupby('member').resid_r_vs_n048base.mean().round(3); print(df.to_string())
# paired FOV bootstrap: E3(X) - E3(control)
B = 500; out = []
for X in ['n053_cytoland', 'n053_dinobloom_b']:
    diffs = np.zeros(B); point = 0
    for d in ('b10', 'b01'):
        base = [L('n048_v9_base', d), L('n048_v9_zoom2', d)]; G = base[0]['G']
        PX = (base[0]['P'] + base[1]['P'] + L(X, d)['P']) / 3; PC = (base[0]['P'] + base[1]['P'] + L('n040_student_base', d)['P']) / 3
        fovs = np.unique(FOV[G]); point += np.mean([r_(PX[M[G, j], j], Y[G[M[G, j]], j]) - r_(PC[M[G, j], j], Y[G[M[G, j]], j]) for j in c11]) / 2
        idx_by_fov = {f: np.where(FOV[G] == f)[0] for f in fovs}
        for b in range(B):
            s = np.concatenate([idx_by_fov[f] for f in rng.choice(fovs, len(fovs))])
            diffs[b] += np.mean([r_(PX[s][M[G[s], j], j], Y[G[s][M[G[s], j]], j]) - r_(PC[s][M[G[s], j], j], Y[G[s][M[G[s], j]], j]) for j in c11]) / 2
    out.append(dict(member=X, diff_vs_r50_third=point, ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5)))
    print(X, 'E3 - E3(r50 third member) =', round(point, 4), 'FOV-bootstrap 95% CI', np.round(np.percentile(diffs, [2.5, 97.5]), 4))
pd.concat([pd.DataFrame(rows), pd.DataFrame(out)]).to_csv(sys.argv[1], index=False)
