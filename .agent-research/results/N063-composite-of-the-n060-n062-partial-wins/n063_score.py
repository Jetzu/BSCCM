"""N063 pre-registered composite: pooled targets (grp:CD3/CD19/CD56, grp:CD45, grp:CD16) from the panel model, all singles from the
twopart + fov-re model; fresh seeds 2-3; vs r50 control seeds 2-3. Secondary: composite (seed-averaged) as 4th member of the N053
ensemble (N048 base + zoom2 + Cytoland). writes n063_scores.csv, n063_per_target.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def rs(P, G): return {names[j]: float(np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1]) for j in c11}
def composite(s, d):
    a, b = L(f'n063_tpfov_s{s}_{d}'), L(f'n063_panel_s{s}_{d}'); assert np.array_equal(a['G'], b['G']); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
rows, pt = [], []
for s in (2, 3):
    for arm in ('composite', 'tpfov', 'panel', 'ctrl'):
        g = []
        for d in ('b10', 'b01'):
            if arm == 'composite': P, G = composite(s, d)
            else: q = L(f'n063_{arm}_s{s}_{d}'); P, G = q['P'], q['G']
            r = rs(P, G); g.append(np.mean(list(r.values())))
            for k, v in r.items(): pt.append(dict(arm=arm, seed=s, direction=d, target=k, r=v))
        rows.append(dict(arm=arm, seed=s, kind='single', goal=float(np.mean(g))))
df = pd.DataFrame(rows); m = df.groupby('arm').goal.mean()
for arm in ('composite', 'ctrl'):
    e = []
    for d in ('b10', 'b01'):
        base = [L(f'{r}_{d}') for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')]; G = base[0]['G']
        X = np.mean([composite(s, d)[0] for s in (2, 3)], 0) if arm == 'composite' else np.mean([L(f'n063_ctrl_s{s}_{d}')['P'] for s in (2, 3)], 0)
        P = (sum(b['P'] for b in base) + X) / 4; e.append(np.mean(list(rs(P, G).values())))
    rows.append(dict(arm=arm, seed=-1, kind='ensemble4_with_N053', goal=float(np.mean(e))))
df = pd.DataFrame(rows); df.to_csv('n063_scores.csv', index=False); pd.DataFrame(pt).to_csv('n063_per_target.csv', index=False)
print(df.round(4).to_string(index=False)); print('seed means:', m.round(4).to_dict()); print('composite - ctrl (pre-registered primary):', round(m['composite'] - m['ctrl'], 4))
