"""N038 D3: GOAL seed noise on v5 labels from existing seed pairs (n023_v5_{b10,b01} seed 0 vs _s1; n026_zoom2 seed 0 vs _s1),
common-11, N031 scorer. Writes d3_seed_noise.json."""
import sys, json, numpy as np
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def sc(r):
    E = np.load(f'{W}/runs/{r}/eval_preds.npz'); P, G = E['P'], E['G']
    return float(np.mean([np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1] for j in c11]))
out = {}
for fam in ('n023_v5', 'n026_zoom2'):
    s0 = {d: sc(f'{fam}_{d}') for d in ('b10', 'b01')}; s1 = {d: sc(f'{fam}_{d}_s1') for d in ('b10', 'b01')}
    out[fam] = dict(s0=s0, s1=s1, goal_s0=np.mean(list(s0.values())), goal_s1=np.mean(list(s1.values())), diff=abs(np.mean(list(s0.values())) - np.mean(list(s1.values()))))
diffs = [v['diff'] for v in out.values()]; out['sd_from_pairs'] = float(np.sqrt(np.mean(np.square(diffs)) / 2))
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1] + '/d3_seed_noise.json', 'w'), indent=1)
