"""N051: existing predictions scored on v10 (and v5) + seed noise on v10 from the existing seed pairs (n023_v5 s0/s1, n026_zoom2 s0/s1).
Writes n051_rescore.csv and n051_noise_v10.json."""
import sys, json, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; out = []
def score(T, runs_by_dir):
    Y, M = T['Y'], T['mask']; names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; g = {}
    for d, runs in runs_by_dir.items():
        Ps = [np.load(f'{W}/runs/{r}/eval_preds.npz') for r in runs]; G = Ps[0]['G']; P = np.mean([p['P'] for p in Ps], 0)
        g[d] = float(np.mean([np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1] for j in c11]))
    return g['b10'], g['b01'], (g['b10'] + g['b01']) / 2
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
sets = {'N031 ensemble': ['n031_base', 'n031_zoom2'], 'N040 students': ['n040_student_base', 'n040_student_zoom2'], 'N048 (trained v9) ensemble': ['n048_v9_base', 'n048_v9_zoom2'], '10-model': fam}
for lab, tf in (('v5', 'targets_v5.npz'), ('v10', 'targets_v5cx_v10.npz')):
    T = np.load(f'{W}/{tf}', allow_pickle=True)
    for k, mem in sets.items():
        b10, b01, g = score(T, {d: [f'{m}_{d}' for m in mem] for d in ('b10', 'b01')}); out.append(dict(labels=lab, model=k, b10=b10, b01=b01, goal=g))
df = pd.DataFrame(out); df.to_csv(sys.argv[1] + '/n051_rescore.csv', index=False); print(df.round(4).to_string(index=False))
T = np.load(f'{W}/targets_v5cx_v10.npz', allow_pickle=True); diffs = {}
for fam_ in ('n023_v5', 'n026_zoom2'):
    a = score(T, {d: [f'{fam_}_{d}'] for d in ('b10', 'b01')})[2]; b = score(T, {d: [f'{fam_}_{d}_s1'] for d in ('b10', 'b01')})[2]; diffs[fam_] = abs(a - b)
sd = float(np.sqrt(np.mean(np.square(list(diffs.values()))) / 2)); json.dump(dict(pair_diffs=diffs, sd=sd), open(sys.argv[1] + '/n051_noise_v10.json', 'w'), indent=1); print('v10 seed noise', diffs, 'sd', round(sd, 4))
