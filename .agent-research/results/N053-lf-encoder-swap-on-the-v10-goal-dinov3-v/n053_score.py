"""N053: single-model and ensemble GOAL (v10 labels, common-11, batch-held-out r) for run families. usage: n053_score.py OUTCSV fam1 [fam2 ...]
A family name f scores runs/f_b10 and runs/f_b01; 'a+b' = ensemble (mean predictions) of families a and b."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; rows = []
def r_dir(fams, d):
    Ps = [np.load(f'{W}/runs/{f}_{d}/eval_preds.npz') for f in fams]; G = Ps[0]['G']
    assert all(np.array_equal(p['G'], G) for p in Ps); P = np.mean([p['P'] for p in Ps], 0)
    return {names[j]: float(np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1]) for j in c11}
for spec in sys.argv[2:]:
    fams = spec.split('+'); per = {d: r_dir(fams, d) for d in ('b10', 'b01')}
    row = dict(model=spec, b10=np.mean(list(per['b10'].values())), b01=np.mean(list(per['b01'].values())))
    row['goal'] = (row['b10'] + row['b01']) / 2
    for d in per:
        for n, v in per[d].items(): row[f'{d}|{n}'] = v
    rows.append(row); print(spec, round(row['b10'], 4), round(row['b01'], 4), 'goal', round(row['goal'], 4), flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1], index=False)
