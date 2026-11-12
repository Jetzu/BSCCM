"""N038 assumption breaker F: (1) seed noise of the GOAL from all v5 seed pairs; (2) per-target FOV-bootstrap SD of r (N031 ensemble);
(3) paired FOV-bootstrap SD of a model difference (N031 ens - N023 single s0); (4) D1b ceiling with the label-version confound removed
(n015 FOV model was trained on v4 labels; n024_r18_v5_fov on v5); (5) composition sensitivity: draw-1 test cells importance-weighted
to draw-0 predicted cls3 composition; (6) r without censored labels per target."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, Y4, M, bt, fov, cz = T['Y'], T['Y_v4'], T['mask'], T['batch'], T['fov'], T['censored']; C11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G'], E['C3']
def goal_dir(P, G, Yv=Y): return np.mean([np.corrcoef(P[M[G, j], j], Yv[G[M[G, j]], j])[0, 1] for j in C11])
print('(1) seed pairs, GOAL = mean of b10/b01 directions')
for base in ('n023_v5', 'n026_zoom2'):
    g = []
    for s in ('', '_s1'):
        g.append(np.mean([goal_dir(*load(f'{base}_{d}{s}')[:2]) for d in ('b10', 'b01')]))
    print(f'  {base}: s0 {g[0]:.4f} s1 {g[1]:.4f} diff {g[1]-g[0]:+.4f}')
rng = np.random.default_rng(1)
def wr(p, y, w): pm, ym = np.average(p, weights=w), np.average(y, weights=w); return np.sum(w * (p - pm) * (y - ym)) / np.sqrt(np.sum(w * (p - pm) ** 2) * np.sum(w * (y - ym) ** 2))
ens = {}; single = {}
for d, b in (('b10', 0), ('b01', 1)):
    A = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; ens[b] = (np.mean([x[0] for x in A], 0), A[0][1], A[0][2]); P1, G1, _ = load(f'n023_v5_{d}'); assert np.array_equal(G1, A[0][1]); single[b] = P1
print('(2)+(3) FOV bootstrap (B=500)')
B = 500; per = {}; diffs = np.zeros(B); goals = np.zeros(B)
fovs = {b: np.unique(fov[ens[b][1]]) for b in (0, 1)}
for i in range(B):
    ge, gs = [], []
    for b in (0, 1):
        P, G, _ = ens[b]; pick = rng.choice(fovs[b], len(fovs[b])); u, c = np.unique(pick, return_counts=True); cnt = pd.Series(c, index=u)
        w_all = cnt.reindex(fov[G]).fillna(0).values
        for j in C11:
            v = M[G, j]; w = w_all[v]; y = Y[G[v], j]; re = wr(P[v, j], y, w); rs = wr(single[b][v, j], y, w); ge.append(re); gs.append(rs); per.setdefault((b, names[j]), []).append(re)
    goals[i] = np.mean(ge); diffs[i] = np.mean(ge) - np.mean(gs)
print(f'  GOAL ens bootstrap sd {goals.std():.4f}; paired diff ens - single: mean {diffs.mean():+.4f} sd {diffs.std():.4f}')
sd = pd.Series({k: np.std(v) for k, v in per.items()}); print('  per-target r bootstrap sd:'); print(sd.round(3).to_string())
print('(4) D1b ceiling, identical cells, per target: n015 (v4-trained r50) vs n024_r18 (v5-trained) FOV models vs N031 ensemble')
Pf, Gf, _ = load('n015_sslinit_fov'); Pr, Gr, _ = load('n024_r18_v5_fov'); assert np.array_equal(Gf, Gr)
rows = []
for b in (0, 1):
    P, G, _ = ens[b]; c = np.intersect1d(Gf[bt[Gf] == b], G); fi = pd.Series(np.arange(len(Gf)), index=Gf)[c].values; ei = pd.Series(np.arange(len(G)), index=G)[c].values
    for j in C11:
        m = M[c, j]; y = Y[c[m], j]
        rows.append(dict(draw=b, target=names[j], n015_v5=np.corrcoef(Pf[fi[m], j], y)[0, 1], n015_vs_v4labels=np.corrcoef(Pf[fi[m], j], Y4[c[m], j])[0, 1],
                         r18v5=np.corrcoef(Pr[fi[m], j], y)[0, 1], ens=np.corrcoef(P[ei[m], j], y)[0, 1]))
d = pd.DataFrame(rows); d['best_fov'] = d[['n015_v5', 'r18v5']].max(1); print(d.round(3).to_string(index=False))
print('  means:', d[['n015_v5', 'r18v5', 'best_fov', 'ens']].mean().round(4).to_dict())
d2 = d.copy(); grp = d2.target.str.startswith('grp:'); d2.loc[grp, 'n015_v5'] = d2.loc[grp, 'r18v5']
print('  n015 with grp targets taken from the v5-trained r18 FOV model:', round(d2.n015_v5.mean(), 4))
d.to_csv(sys.argv[1] + '/f_d1b_ceiling.csv', index=False)
print('(5) composition: draw-1 test cells reweighted to draw-0 predicted cls3 composition (per slide; C3 from the v5 FOV r18 model)')
_, _, C3f = load('n024_r18_v5_fov'); c3f = pd.Series(C3f, index=Gf)
for j in C11:
    P, G, C3 = ens[1]; v = M[G, j]; g1 = G[v]; k1 = c3f.reindex(g1).values
    g0 = Gf[(bt[Gf] == 0) & M[Gf, j]]; k0 = c3f.reindex(g0).values
    if np.any(np.isnan(k1)): ok = ~np.isnan(k1)
    else: ok = np.ones(len(g1), bool)
    f1 = np.bincount(k1[ok].astype(int), minlength=3) / ok.sum(); f0 = np.bincount(k0.astype(int), minlength=3) / len(k0)
    w = (f0 / np.maximum(f1, 1e-9))[k1[ok].astype(int)]; p = P[v, j][ok]; y = Y[g1[ok], j]
    print(f'  {names[j]:20s} n {ok.sum():5d} comp d1 {np.round(f1,2)} d0 {np.round(f0,2)}  r {np.corrcoef(p, y)[0,1]:.3f} -> reweighted {wr(p, y, w):.3f}')
print('(6) r without censored labels')
for b in (0, 1):
    P, G, _ = ens[b]
    for j in C11:
        v = M[G, j]; c = cz[G[v], j]
        if c.mean() > 0.05: print(f'  draw {b} {names[j]:20s} censored {c.mean():.2f}  r {np.corrcoef(P[v, j], Y[G[v], j])[0,1]:.3f}  r_noncens {np.corrcoef(P[v, j][~c], Y[G[v], j][~c])[0,1]:.3f}  label sd cens {Y[G[v], j][c].std():.2f} vs non {Y[G[v], j][~c].std():.2f}')
