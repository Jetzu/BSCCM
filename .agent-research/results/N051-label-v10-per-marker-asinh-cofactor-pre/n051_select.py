"""N051 pre-registered per-marker cofactor selection (label-only). Candidates q10 (v5cx), q50, 15, 50, 150, 500 (targets_v5cx[_cf*].npz).
Per single marker: (a) split-0 FOV excess (vs 20-perm null, FOVs >= 3 cells), mean over draws; (b) tail contrast = (mean of top 0.5 % -
median) / (1.4826 MAD) on split-0, mean over draws; lineage r(cand, v5) per draw. RULE: lowest (a) s.t. (b) >= .5 x (b)[q10] and, for
CD16/CD45/CD14, r >= .95 in both draws; ties within .005 in (a) -> larger cofactor. Writes n051_metrics.csv, n051_selection.json and
prints the BSCCM_COFACTOR map string on the last line."""
import sys, json, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; rng = np.random.default_rng(0)
C = {'q10': 'targets_v5cx.npz', 'q50': 'targets_v5cx_cfq50.npz', '15': 'targets_v5cx_cf15.npz', '50': 'targets_v5cx_cf50.npz', '150': 'targets_v5cx_cf150.npz', '500': 'targets_v5cx_cf500.npz'}
ORDER = {'q10': 0, 'q50': 0.5, '15': 15, '50': 50, '150': 150, '500': 500}   # 'larger cofactor' tie-break (per-slide ones ranked lowest)
V = {k: np.load(f'{W}/{f}', allow_pickle=True) for k, f in C.items()}; V5 = np.load(W + '/targets_v5.npz', allow_pickle=True)
names = [str(n) for n in V5['names']]; fov, bt, sp, hr = V5['fov'], V5['batch'], V5['split'], V5['hist_row']
def excess(y, f):
    u, inv, c = np.unique(f, return_inverse=True, return_counts=True); k = c[inv] >= 3; y, f = y[k], f[k]; u, inv = np.unique(f, return_inverse=True); n = np.bincount(inv)
    s = lambda v: np.var((np.bincount(inv, v) / n)[inv]) / np.var(v); return float(s(y) - np.mean([s(rng.permutation(y)) for _ in range(20)]))
def tail(y):
    med = np.median(y); mad = 1.4826 * np.median(np.abs(y - med)) + 1e-9; top = np.sort(y)[-max(1, int(.005 * len(y))):]; return float((top.mean() - med) / mad)
rows = []
for m in ['CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3']:
    j = names.index(f'single:{m}')
    for k, T in V.items():
        e, t, r = [], [], []
        for b in (0, 1):
            s = T['mask'][:, j] & (bt == b) & (hr < 0); s0 = s & (sp == 0); y = T['Y'][s0, j].astype(float)
            e.append(excess(y, fov[s0])); t.append(tail(y)); both = s & V5['mask'][:, j]; r.append(float(np.corrcoef(T['Y'][both, j], V5['Y'][both, j])[0, 1]))
        rows.append(dict(marker=m, cofactor=k, excess=np.mean(e), excess_d1=e[0], excess_d2=e[1], tail=np.mean(t), r_v5_min=min(r)))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n051_metrics.csv', index=False); pd.set_option('display.width', 200)
print(df.pivot_table(index='marker', columns='cofactor', values=['excess', 'tail', 'r_v5_min']).round(3).to_string())
sel = {}
for m, g in df.groupby('marker'):
    t0 = float(g[g.cofactor == 'q10']['tail'].iloc[0]); ok = g[(g['tail'] >= .5 * t0) & ((g.r_v5_min >= .95) | ~g.marker.isin(['CD16', 'CD45', 'CD14']))]
    if ok.empty: sel[m] = 'q10'; continue
    best = ok.excess.min(); tied = ok[ok.excess <= best + .005]; sel[m] = max(tied.cofactor, key=lambda c: ORDER[c])
json.dump(dict(selection=sel, rule=__doc__), open(sys.argv[1] + '/n051_selection.json', 'w'), indent=1)
print('SELECTION', sel); print('map:' + ','.join(f'{m}={c}' for m, c in sel.items()))
