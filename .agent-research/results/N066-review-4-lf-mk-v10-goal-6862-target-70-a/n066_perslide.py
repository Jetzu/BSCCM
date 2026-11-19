"""N066 diagnosis: per test-slide Pearson r of the 6-way ensemble (N065 E3c + N053 members) for every common-11 target,
plus the pooled-over-slides r and an 'equalised' r (per-slide mean/sd matched to the pooled label) to separate between-slide
offset error from within-slide ranking. writes n066_perslide.csv"""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M, S = T['Y'], T['mask'], T['slide']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp, pa, d):
    a, b = L(f'{tp}_{d}'), L(f'{pa}_{d}'); P = a['P'].copy(); P[:, pooled] = b['P'][:, pooled]; return P, a['G']
rows = []
for d in ('b10', 'b01'):
    base = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    z2, G = comp('n065_tpfov_z2', 'n065_panel_z2', d); cy, _ = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)
    P = np.mean([base, z2, cy] + [L(f'{r}_{d}')['P'] for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')], 0)
    for j in c11:
        m = M[G, j]; p, y, s = P[m, j], Y[G[m], j], S[G[m]]
        rows.append(dict(direction=d, target=names[j], slide='ALL', n=int(m.sum()), r=np.corrcoef(p, y)[0, 1],
                         ymean=y.mean(), pmean=p.mean(), ysd=y.std(), psd=p.std()))
        for sl in np.unique(s):
            k = s == sl
            if k.sum() < 200: continue
            rows.append(dict(direction=d, target=names[j], slide=str(T['slide_names'][sl]), n=int(k.sum()), r=np.corrcoef(p[k], y[k])[0, 1],
                             ymean=y[k].mean(), pmean=p[k].mean(), ysd=y[k].std(), psd=p[k].std()))
df = pd.DataFrame(rows); df.to_csv('n066_perslide.csv', index=False)
pd.set_option('display.width', 200); print(df.round(3).to_string(index=False))
