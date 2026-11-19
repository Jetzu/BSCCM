"""N066 diagnosis: 6-way ensemble (N065) scored against v10 labels and against the N064 local-corner-background reconstruction
('full', same pipeline otherwise) on the same held-out cells. If predictions agree MORE with the alternative background labels on
targets where v10 and local disagree, the v10 background model (not the image model) limits GOAL. writes n066_altlabels.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
H = np.load(W + '/runs/n064_ceiling/half_labels.npz'); hn = [str(n) for n in H['names']]; row = {g: i for i, g in enumerate(H['G'])}
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(f'{tp}_{d}'), L(f'{pa}_{d}'); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
rows = []
for d in ('b10', 'b01'):
    base = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    z2, G = comp('n065_tpfov_z2', 'n065_panel_z2', d); cy, _ = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)
    P = np.mean([base, z2, cy] + [L(f'{r}_{d}')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')], 0)
    for j in c11:
        m = M[G, j] & np.array([g in row for g in G]); g = G[m]; hi = np.array([row[x] for x in g]); k = hn.index(names[j])
        yl = H['full'][hi, k]; ok = np.isfinite(yl); p, y, yl = P[m, j][ok], Y[g, j][ok], yl[ok]
        z = lambda v: (v - v.mean()) / v.std(); yc = z(y) + z(yl)
        rows.append(dict(direction=d, target=names[j], n=int(ok.sum()), n_mask=int(M[G, j].sum()), r_v10=np.corrcoef(p, y)[0, 1],
                         r_local=np.corrcoef(p, yl)[0, 1], r_consensus=np.corrcoef(p, yc)[0, 1], r_v10_local=np.corrcoef(y, yl)[0, 1]))
df = pd.DataFrame(rows); df.to_csv('n066_altlabels.csv', index=False); print(df.round(3).to_string(index=False))
print(df.groupby('direction')[['r_v10', 'r_local', 'r_consensus']].mean().round(4)); print(df[['r_v10', 'r_local', 'r_consensus']].mean().round(4))
