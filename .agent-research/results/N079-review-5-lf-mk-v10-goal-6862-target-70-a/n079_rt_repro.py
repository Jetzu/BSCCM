"""N079: first-hand reproduction of the Codex red-team's cell-type-only control (b01): each held-out 'all' cell predicted by the FOV-cross-
fitted mean pooled label of its PREDICTED 3-class type (C3 from n065_tpfov_z2_b01) vs composed single outputs (n079_joint_consistency)."""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, F = T['Y'], T['mask'], T['fov']
names = [str(n) for n in T['names']]; ix = {n: i for i, n in enumerate(names)}
z = np.load(f'{W}/runs/n065_tpfov_z2_b01/eval_preds.npz'); G, C3 = z['G'], z['C3']
rows = []
for g in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16'):
    j = ix[g]; m = M[G, j]; y = Y[G[m], j]; c = C3[m]; fov = F[G[m]]
    u = np.unique(fov); rng = np.random.default_rng(0); fold = dict(zip(u, rng.integers(0, 5, len(u)))); f = np.array([fold[v] for v in fov]); pred = np.zeros_like(y)
    for i in range(5):
        tr, te = f != i, f == i
        means = {k: y[tr & (c == k)].mean() if (tr & (c == k)).any() else y[tr].mean() for k in range(3)}; pred[te] = [means[k] for k in c[te]]
    rows.append(dict(pooled=g, r_celltype_only=np.corrcoef(pred, y)[0, 1]))
df = pd.DataFrame(rows); jc = pd.read_csv('n079_joint_consistency.csv'); jc = jc[jc.direction == 'b01'].set_index('pooled')
df['r_composed_singles'] = df.pooled.map(jc.r_composed_singles); df['r_pooled_head'] = df.pooled.map(jc.r_pooled_head)
print(df.round(4).to_string(index=False)); print(df.drop(columns='pooled').mean().round(4).to_string()); df.to_csv('n079_rt_repro.csv', index=False)
