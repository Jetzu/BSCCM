"""N038 diagnosis D1: on IDENTICAL test cells (held-out FOVs of each draw), in-distribution FOV-split model (n024_r18_v5_fov: r18,
trained on split 0 of BOTH draws) vs the batch-held-out GOAL ensemble (n031 base+zoom2, trained on the OTHER draw only).
Per target x draw Pearson r on v5 labels. A large FOV>batch gap with a WEAKER model = the draw shift, not model capacity, bounds the GOAL."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, bt, sp = T['Y'], T['mask'], T['batch'], T['split']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
Pf, Gf = load('n024_r18_v5_fov'); print('fov eval cells', len(Gf), 'splits', np.unique(sp[Gf]), 'batches', np.unique(bt[Gf], return_counts=True))
rows = []
for d, test_b in (('b10', 0), ('b01', 1)):
    Ps = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; Gb = Ps[0][1]; Pb = np.mean([p[0] for p in Ps], 0)
    common = np.intersect1d(Gf[bt[Gf] == test_b], Gb); fi = pd.Series(np.arange(len(Gf)), index=Gf)[common].values; bi = pd.Series(np.arange(len(Gb)), index=Gb)[common].values
    for j in c11:
        m = M[common, j]
        if m.sum() < 100: continue
        y = Y[common[m], j]
        rows.append(dict(test_draw=test_b, target=names[j], n=int(m.sum()), r_fov_r18=float(np.corrcoef(Pf[fi[m], j], y)[0, 1]), r_batch_ens=float(np.corrcoef(Pb[bi[m], j], y)[0, 1])))
df = pd.DataFrame(rows); df['gap'] = df.r_fov_r18 - df.r_batch_ens; df.to_csv(sys.argv[1] + '/d1_fov_vs_batch.csv', index=False)
pd.set_option('display.width', 200); print(df.round(3).to_string(index=False)); print(df.groupby('test_draw')[['r_fov_r18', 'r_batch_ens', 'gap']].mean().round(4))
