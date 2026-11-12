"""N038 diagnosis D1b: same architecture + same SSL encoder (n007 r50, 10 ep): FOV split trained on BOTH draws (n015_sslinit_fov,
v4 training labels) vs batch-held-out trained on the other draw (n023_v5_{b10,b01}, v5 labels), identical test cells (split 2 of the
test draw), v5 labels. Gap = cost of never seeing the test draw (draw shift + half the data)."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M, bt = T['Y'], T['mask'], T['batch']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
Pf, Gf = load('n015_sslinit_fov'); rows = []
for d, b in (('b10', 0), ('b01', 1)):
    Pb, Gb = load(f'n023_v5_{d}'); c = np.intersect1d(Gf[bt[Gf] == b], Gb)
    fi = pd.Series(np.arange(len(Gf)), index=Gf)[c].values; bi = pd.Series(np.arange(len(Gb)), index=Gb)[c].values
    for j in c11:
        m = M[c, j]; y = Y[c[m], j]
        rows.append(dict(test_draw=b, target=names[j], n=int(m.sum()), r_fov=float(np.corrcoef(Pf[fi[m], j], y)[0, 1]), r_batch=float(np.corrcoef(Pb[bi[m], j], y)[0, 1])))
df = pd.DataFrame(rows); df['gap'] = df.r_fov - df.r_batch; df.to_csv(sys.argv[1] + '/d1b_same_arch.csv', index=False)
pd.set_option('display.width', 200); print(df.round(3).to_string(index=False)); print(df.groupby('test_draw')[['r_fov', 'r_batch', 'gap']].mean().round(4)); print('overall', df[['r_fov', 'r_batch', 'gap']].mean().round(4).to_dict())
