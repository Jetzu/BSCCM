"""N048 pre-registered label QC (label-only, no model predictions): per cofactor candidate (q10 = v5cx, q50, 50, 150), single markers x
draws: training-split (split 0) FOV excess share (vs 20-perm null, FOVs >= 3 cells) and r(candidate, v5) per marker x draw.
Selection rule: lowest mean split-0 FOV excess over 8 markers x 2 draws, subject to r >= .95 for CD16/CD45/CD14 and no marker's excess
rising by > .05 vs q10. Usage: n048_qc.py OUTDIR"""
import sys, json, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; rng = np.random.default_rng(0)
F = {'q10': 'targets_v5cx.npz', 'q50': 'targets_v5cx_cfq50.npz', '50': 'targets_v5cx_cf50.npz', '150': 'targets_v5cx_cf150.npz'}
V = {k: np.load(f'{W}/{f}', allow_pickle=True) for k, f in F.items()}; V5 = np.load(W + '/targets_v5.npz', allow_pickle=True)
names = [str(n) for n in V5['names']]; fov, bt, sp, hr = V5['fov'], V5['batch'], V5['split'], V5['hist_row']
def excess(y, f):
    u, inv, c = np.unique(f, return_inverse=True, return_counts=True); k = c[inv] >= 3; y, f = y[k], f[k]; u, inv = np.unique(f, return_inverse=True); n = np.bincount(inv)
    s = lambda v: np.var((np.bincount(inv, v) / n)[inv]) / np.var(v); return float(s(y) - np.mean([s(rng.permutation(y)) for _ in range(20)]))
rows = []
for k, T in V.items():
    for m in ['CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3']:
        j = names.index(f'single:{m}')
        for b in (0, 1):
            s = T['mask'][:, j] & (bt == b) & (hr < 0); s0 = s & (sp == 0)
            rows.append(dict(cofactor=k, marker=m, draw=b, excess_split0=excess(T['Y'][s0, j].astype(float), fov[s0]),
                             r_vs_v5=float(np.corrcoef(T['Y'][s & V5['mask'][:, j], j], V5['Y'][s & V5['mask'][:, j], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n048_label_qc.csv', index=False); pd.set_option('display.width', 200)
print(df.pivot_table(index='marker', columns=['cofactor', 'draw'], values='excess_split0').round(3).to_string())
print(df.pivot_table(index='marker', columns=['cofactor', 'draw'], values='r_vs_v5').round(3).to_string())
base = df[df.cofactor == 'q10'].set_index(['marker', 'draw']).excess_split0; res = {}
for k in ['q50', '50', '150']:
    d = df[df.cofactor == k].set_index(['marker', 'draw'])
    lin_ok = bool((d.loc[['CD16', 'CD45', 'CD14'], 'r_vs_v5'] >= .95).all()); rise_ok = bool(((d.excess_split0 - base) <= .05).all())
    res[k] = dict(mean_excess=float(d.excess_split0.mean()), lineage_r_ok=lin_ok, no_rise_ok=rise_ok, passes=lin_ok and rise_ok)
res['q10'] = dict(mean_excess=float(base.mean())); ok = {k: v for k, v in res.items() if v.get('passes')}
res['SELECTED'] = min(ok, key=lambda k: ok[k]['mean_excess']) if ok else None
print(json.dumps(res, indent=1)); json.dump(res, open(sys.argv[1] + '/n048_selection.json', 'w'), indent=1)
