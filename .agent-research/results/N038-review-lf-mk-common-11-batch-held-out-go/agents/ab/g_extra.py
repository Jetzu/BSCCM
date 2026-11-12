"""N038 assumption breaker G: (1) censored labels collapsed to their (target x draw) mean -> r (does the detrend of floor values cost r?);
(2) HLA-DR draw-1 deficit (held-out ens vs in-distribution FOV models): per predicted class (C3 of the v5 r18 FOV model) label/pred means and
within-class r -> label-semantics shift (dim d1 dye) vs image shift; (3) share of GOAL test-sampling variance from CD123 + CD19."""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt, cz = T['Y'], T['mask'], T['batch'], T['censored']; C11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G'], E['C3']
ens = {}
for d, b in (('b10', 0), ('b01', 1)):
    A = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; ens[b] = (np.mean([x[0] for x in A], 0), A[0][1])
tot, tot2 = [], []
for b in (0, 1):
    P, G = ens[b]
    for j in C11:
        v = M[G, j]; y = Y[G[v], j].copy(); c = cz[G[v], j]; r0 = np.corrcoef(P[v, j], y)[0, 1]
        if c.any(): y[c] = y[c].mean()
        r1 = np.corrcoef(P[v, j], y)[0, 1]; tot.append(r0); tot2.append(r1)
        if c.mean() > .05: print(f'draw {b} {names[j]:18s} cens {c.mean():.2f} r {r0:.3f} -> censored collapsed {r1:.3f}')
print(f'GOAL {np.mean(tot):.4f} -> with censored test labels collapsed {np.mean(tot2):.4f}')
Pf, Gf, C3f = load('n024_r18_v5_fov'); Pn, Gn, _ = load('n015_sslinit_fov'); c3 = pd.Series(C3f, index=Gf); pf = pd.DataFrame(Pf, index=Gf); pn = pd.DataFrame(Pn, index=Gn)
j = names.index('single:HLA-DR')
for b in (0, 1):
    P, G = ens[b]; v = M[G, j]; g = G[v]; ok = np.isin(g, Gf); g = g[ok]; pe = P[v, j][ok]; y = Y[g, j]; k = c3.reindex(g).values; p1 = pf.loc[g, j].values; p2 = pn.loc[g, j].values
    print(f'HLA-DR draw {b}: n {len(g)} r ens {np.corrcoef(pe, y)[0,1]:.3f} r18fov {np.corrcoef(p1, y)[0,1]:.3f} n015fov {np.corrcoef(p2, y)[0,1]:.3f}')
    zs = lambda x: (x - x.mean()) / x.std()
    for kk, kn in enumerate(['lym', 'gran', 'mono']):
        s = k == kk
        print(f'   {kn:4s} frac {s.mean():.2f}  label mean {y[s].mean():+.2f}  pred z-mean ens {zs(pe)[s].mean():+.2f} r18 {zs(p1)[s].mean():+.2f} n015 {zs(p2)[s].mean():+.2f} | within r ens {np.corrcoef(pe[s], y[s])[0,1]:.3f} r18 {np.corrcoef(p1[s], y[s])[0,1]:.3f} n015 {np.corrcoef(p2[s], y[s])[0,1]:.3f}')
# (3) variance shares from the f_extra bootstrap SDs (printed there): CD123 .098/.072, CD19 .040/.090, total GOAL sd .0081
s = {'CD123': [.098, .072], 'CD19': [.040, .090]}; tv = .0081 ** 2
for k, v in s.items(): print(k, 'share of GOAL test-sampling variance (independence approx):', round(sum(x * x for x in v) / 22 ** 2 / tv, 2))
