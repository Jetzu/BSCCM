"""N034-B (scalar bench of the user's set-level idea): can a marker readout be learnt from SET-level distribution matching alone, with
the cell-level pairing hidden? Features = frozen self-supervised LF embedding (n007 SSL(aug) r50, no labels used) -> 64 PCs; label =
curated targets_v5 marker of a single-antibody stain (or autofl on unstained slides). Train: blood draw 1 split 0; test: draw 0 (new
draw) and draw 1 split 2 (held-out FOVs).
  paired : ridge on true pairs (reference).
  same   : bags of n cells of one slide, neighbouring FOVs (sorted by FOV); the bag's label MULTISET is known, not which cell has which.
  other  : LF bag and label bag of n DIFFERENT cells of the same slide (= the Giemsa situation: same blood draw, different cells).
Learner (1-D optimal transport, iterated): p = Xw; within each bag assign the sorted labels to the cells in order of p; refit ridge;
30 iterations x 8 random starts; the start with the lowest set-level objective (mean within-bag squared distance between sorted p
and sorted labels -- uses bag information only) is kept. Writes CSV/JSON to --out.
v2 (red-team N034 rereview N3-N5): clean targets_v5c; new draw scored on split 2 only; learners OT (above), BAGMEAN (ridge of bag-mean
label on bag-mean features, uses only the label multiset's mean) and OT initialised from BAGMEAN; bag designs matched for budget and
locality: paired / paired_A (half A only), same_fov (FOV-sorted bags), same_rand (random bags), same_A (half A, random), other (LF from
half A, labels from half B, random), other_fov (A and B cells of the SAME FOV-group: different cells, shared spatial stratum).
Features: n007 SSL(aug) -- unlabelled pretraining that saw test-draw images (transductive, N031 showed no GOAL inflation)."""
import os; os.environ.setdefault('OMP_NUM_THREADS', '4')
import json, argparse, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--n-train', type=int, default=20000)
ap.add_argument('--sizes', default='1,4,16,64,256,2048'); ap.add_argument('--starts', type=int, default=8); ap.add_argument('--iters', type=int, default=30)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
W = '/hnvme/workspace/v181bb10-bsccm'
T = np.load(W + '/targets_v5c.npz', allow_pickle=True); names = [str(n) for n in T['names']]
FE = np.load(W + '/runs/n007_ssl_aug/probe_fov/lf_feats_all.npy', mmap_mode='r')
stain = np.load(W + '/fm_prep.npz')['stain'].astype(int); SN = [str(s) for s in np.load(W + '/fm_prep.npz')['stain_names']]
clean = T['hist_row'] < 0; fov, slide = T['fov'], T['slide']; rng = np.random.default_rng(0)
pear = lambda x, y: float(np.corrcoef(x, y)[0, 1])
def ridge(X, y, lam=10.):
    A = np.c_[X, np.ones(len(X))]; return np.linalg.solve(A.T @ A + lam * np.diag(np.r_[np.ones(X.shape[1]), 0]), A.T @ y)
pred = lambda X, w: np.c_[X, np.ones(len(X))] @ w

HALF = {}   # fixed random half A / half B split of the training cells per target (budget-matched arms)
def halves(rows):
    k = tuple(rows[:3])
    if k not in HALF:
        pr = np.random.default_rng(7).permutation(len(rows)); HALF[k] = (np.sort(rows[pr[:len(rows) // 2]]), np.sort(rows[pr[len(rows) // 2:]]))
    return HALF[k]
def chunks(r, n): return [r[i:i + n] for i in range(0, len(r) - n + 1, n)]
def bags(rows, n, design):
    out = []; A, B = halves(rows)
    for s in np.unique(slide[rows]):
        r = rows[slide[rows] == s]; a, b = A[slide[A] == s], B[slide[B] == s]
        if design == 'same_fov': r = r[np.lexsort((rng.random(len(r)), fov[r]))]; out += [(c, c) for c in chunks(r, n)]
        elif design == 'same_rand': out += [(c, c) for c in chunks(rng.permutation(r), n)]
        elif design == 'same_A': out += [(c, c) for c in chunks(rng.permutation(a), n)]
        elif design == 'other': out += list(zip(chunks(rng.permutation(a), n), chunks(rng.permutation(b), n)))
        elif design == 'other_fov':   # consecutive FOV-sorted groups of 2n cells, split at random into n LF cells and n label cells
            r = r[np.lexsort((rng.random(len(r)), fov[r]))]
            for c in chunks(r, 2 * n):
                c = rng.permutation(c); out.append((c[:n], c[n:]))
    return out

def bagmean(X, y, bags):
    li = np.stack([b[0] for b in bags]); lj = np.stack([b[1] for b in bags]); n = li.shape[1]
    return ridge(X[li].mean(1), y[lj].mean(1), lam=10. / n)
def set_learn(X, y, bags, init=None):   # bags: list of (lf_rows, label_rows) as row indices into X / y
    best = None
    li = np.stack([b[0] for b in bags]); yi = np.sort(y[np.stack([b[1] for b in bags])], 1); allx = li.ravel()   # equal-size bags
    for s in range(a.starts if init is None else 1):
        w = np.r_[np.random.default_rng(s).standard_normal(X.shape[1]), 0.] if init is None else init.copy()
        for it in range(a.iters):
            p = pred(X, w); ya = np.empty(len(X))
            ya[np.take_along_axis(li, np.argsort(p[li], 1), 1)] = yi
            w = ridge(X[allx], ya[allx])
        p = pred(X, w); obj = float(np.mean((np.sort(p[li], 1) - yi) ** 2))
        if best is None or obj < best[0]: best = (obj, w)
    return best

rows_out = []; t0 = time.time()
for lab, st in [('single:CD16', 'CD16'), ('single:CD14', 'CD14'), ('single:CD45', 'CD45'), ('single:CD3', 'CD3'), ('single:HLA-DR', 'HLA-DR'),
                ('single:CD19', 'CD19'), ('autofl', 'unstained')]:
    j = names.index(lab); base = clean & T['mask'][:, j] & (stain == SN.index(st))
    tr = np.flatnonzero(base & (T['batch'] == 1) & (T['split'] == 0))
    tr = np.sort(rng.choice(tr, min(a.n_train, len(tr)), replace=False))
    te = {'newdraw': np.flatnonzero(base & (T['batch'] == 0) & (T['split'] == 2)), 'inbatch': np.flatnonzero(base & (T['batch'] == 1) & (T['split'] == 2))}
    Xtr = np.asarray(FE[tr], np.float32); mu = Xtr.mean(0); U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False); P = Vt[:64].T
    sd = ((Xtr - mu) @ P).std(0); f = lambda X: ((X - mu) @ P) / sd
    X = f(Xtr); y = T['Y'][tr, j].astype(np.float64); idx = pd.Series(np.arange(len(tr)), index=tr)
    Xte = {k: f(np.asarray(FE[v], np.float32)) for k, v in te.items()}; yte = {k: T['Y'][v, j] for k, v in te.items()}
    A_, _ = halves(tr); ia = np.searchsorted(tr, A_)
    for mode, w in (('paired', ridge(X, y)), ('paired_A', ridge(X[ia], y[ia]))):
        for k in te: rows_out.append(dict(target=lab, mode=mode, learner='paired', n=1, test=k, r=pear(pred(Xte[k], w), yte[k])))
    for mode in ('same_fov', 'same_rand', 'same_A', 'other', 'other_fov'):
        for n in [int(s) for s in a.sizes.split(',')]:
            bg = bags(tr, n, mode)
            if len(bg) < 3: continue
            lb = [(idx[b0].values, idx[b1].values) for b0, b1 in bg]; wb = bagmean(X, y, lb)
            fits = {'bagmean': (np.nan, wb), 'ot': set_learn(X, y, lb), 'ot_init': set_learn(X, y, lb, init=wb)}
            for ln, (obj, w) in fits.items():
                for k in te:
                    rows_out.append(dict(target=lab, mode=mode, learner=ln, n=n, test=k, r=pear(pred(Xte[k], w), yte[k]), objective=obj, n_bags=len(bg),
                                         lf_cells=int(sum(len(b[0]) for b in bg))))
        print(lab, mode, f'{time.time() - t0:.0f}s', flush=True)
df = pd.DataFrame(rows_out); df.to_csv(a.out + '/setmatch_sim_v2.csv', index=False)
pd.set_option('display.width', 220)
print(df[df.test == 'newdraw'].pivot_table(index=['mode', 'learner'], columns='n', values='r', aggfunc='mean').round(3))
