"""N079 diagnosis: joint (multi-marker) decoding consistency on held-out 'all'-slide cells, frozen 6-way ensemble (N065 composite members
+ N053 members). Each pooled measurement is predicted (a) by the model's own pooled head and (b) by a linear combination of the model's
SINGLE-marker outputs on the same cell (5-fold FOV cross-fitted least squares on the held-out cells; diagnostic only). If (b) ~ (a), the
single heads decode a jointly consistent per-cell profile. Also (c) singles-only combination including all 8 single outputs.
writes n079_joint_consistency.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, F = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; ix = {n: i for i, n in enumerate(names)}
pooled = [ix[n] for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(f'{tp}_{d}'), L(f'{pa}_{d}'); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
SPEC = {'grp:CD3/CD19/CD56': ['single:CD3', 'single:CD19', 'single:CD56'], 'grp:CD45': ['single:CD45'], 'grp:CD16': ['single:CD16']}
SING = [n for n in names if n.startswith('single:')]
def cv_lin(X, y, fov, k=5):
    out = np.zeros_like(y); u = np.unique(fov); rng = np.random.default_rng(0); fold = dict(zip(u, rng.integers(0, k, len(u)))); f = np.array([fold[v] for v in fov])
    for i in range(k):
        tr, te = f != i, f == i; A = np.c_[X[tr], np.ones(tr.sum())]; w = np.linalg.lstsq(A, y[tr], rcond=None)[0]; out[te] = np.c_[X[te], np.ones(te.sum())] @ w
    return out
rows = []
for d in ('b10', 'b01'):
    base = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    z2, G = comp('n065_tpfov_z2', 'n065_panel_z2', d); cy, _ = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)
    P = np.mean([base, z2, cy] + [L(f'{r}_{d}')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')], 0)
    for g, sl in SPEC.items():
        j = ix[g]; m = M[G, j]; y = Y[G[m], j]; fov = F[G[m]]
        r_head = np.corrcoef(P[m, j], y)[0, 1]
        r_sing = np.corrcoef(cv_lin(P[m][:, [ix[s] for s in sl]], y, fov), y)[0, 1]
        r_all8 = np.corrcoef(cv_lin(P[m][:, [ix[s] for s in SING]], y, fov), y)[0, 1]
        r_both = np.corrcoef(cv_lin(np.c_[P[m][:, [ix[s] for s in SING]], P[m, j]], y, fov), y)[0, 1]
        rows.append(dict(direction=d, pooled=g, n=int(m.sum()), r_pooled_head=r_head, r_composed_singles=r_sing, r_composed_all8=r_all8, r_head_plus_singles=r_both))
df = pd.DataFrame(rows); df.to_csv('n079_joint_consistency.csv', index=False); print(df.round(3).to_string(index=False)); print(df.drop(columns=['direction', 'pooled', 'n']).mean().round(4).to_string())
