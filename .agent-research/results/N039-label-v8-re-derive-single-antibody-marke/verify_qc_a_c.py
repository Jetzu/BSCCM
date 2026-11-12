"""N039: Claude first-hand check of Codex QC (a) FOV excess and (c) r(v8, v5) on the rejected candidate targets_v8cx.npz (in /tmp)."""
import sys, numpy as np
W = '/hnvme/workspace/v181bb10-bsccm'; V8 = np.load(sys.argv[1], allow_pickle=True); V5 = np.load(W + '/targets_v5.npz', allow_pickle=True)
names = [str(n) for n in V5['names']]; fov, bt, hr = V5['fov'], V5['batch'], V5['hist_row']; rng = np.random.default_rng(0)
def share(y, f):
    u, inv, c = np.unique(f, return_inverse=True, return_counts=True); ok = c[inv] >= 3; y, f = y[ok], f[ok]
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv)
    s = lambda v: np.var((np.bincount(inv, v) / c)[inv]) / np.var(v); s0 = s(y); nul = np.mean([s(rng.permutation(y)) for _ in range(20)]); return (s0 - nul) / (1 - nul)
for m in ('CD19', 'CD56', 'CD3', 'CD14', 'CD16', 'CD45', 'CD123', 'HLA-DR'):
    j = names.index(f'single:{m}')
    for b in (0, 1):
        k = V5['mask'][:, j] & (bt == b) & (hr < 0) & V8['mask'][:, j]
        print(f'{m:7s} draw {b} n {k.sum():6d} fov_excess v8 {share(V8["Y"][k, j].astype(float), fov[k]):.3f} v5 {share(V5["Y"][k, j].astype(float), fov[k]):.3f}  r(v8,v5) {np.corrcoef(V8["Y"][k, j], V5["Y"][k, j])[0, 1]:.3f}')
