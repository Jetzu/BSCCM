"""N038 diagnosis D1c (red-team reproduction): all saved FOV-split models on the GOAL test cells (split 2 of each draw), v5 labels,
common-11, equal-draw mean; plus the GOAL ensemble on the same cells. Reproduces Codex diag_rt 'N022 tail05 .680'."""
import sys, json, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, bt = T['Y'], T['mask'], T['batch']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
rows = []
for run in ['n015_sslinit_fov', 'n022_base_fov', 'n022_tail05_fov', 'n022_tail10_b10', 'n022_tail05_b10', 'n022_base_b10', 'n024_r18_v5_fov']:
    try: P, G = load(run)
    except Exception as e: print(run, 'missing', e); continue
    a = json.load(open(f'{W}/runs/{run}/result.json'))['args']
    for b in (0, 1):
        k = bt[G] == b
        if k.sum() == 0: continue
        for j in c11:
            m = M[G[k], j]; rows.append(dict(run=run, protocol=a['protocol'], tail=a.get('tail_weight'), focal=a.get('focal_c10'), targets=a['targets'], draw=b, target=names[j], r=float(np.corrcoef(P[k][m, j], Y[G[k][m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/d1c_fov_models.csv', index=False)
g = df.groupby(['run', 'protocol', 'tail', 'focal', 'targets', 'draw']).r.mean().unstack('draw'); g['mean'] = g.mean(1); print(g.round(4).to_string())
print(df[df.run.isin(['n022_base_fov', 'n022_tail05_fov'])].pivot_table(index='target', columns=['run', 'draw'], values='r').round(3).to_string())
