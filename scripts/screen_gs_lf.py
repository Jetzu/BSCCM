"""Zero-training GS<->LF alignment screen on FROZEN features (N008). Answers: how much do frozen Wright (DinoBloom) and LF embeddings
share, and does a rotation (Procrustes) suffice vs learned maps, as a function of the number of pairs?
Protocol: fit on GS-LF pairs of the batch-0 stained slide, test on the batch-1 stained slide (held-out slide/isolation), plus a
within-slide FOV split. Methods per n pairs: Procrustes (orthogonal, after per-modality PCA-whitening fit on ALL cells of that
modality = unsupervised), ridge GS->LF, small MLP, linear SigLIP heads. Metrics: GS->LF and LF->GS R@1/R@10/median rank among test
cells; mutual-kNN (k=10, PRH metric) of the raw spaces on test cells.
usage: screen_gs_lf.py --gs emb/giemsa_dinobloom_B.npy --lf <dir with lf_feats_hist.npy + lf_idx_hist.npy> --out DIR"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
ap = argparse.ArgumentParser(); ap.add_argument('--gs', required=True); ap.add_argument('--lf', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--k', type=int, default=128); ap.add_argument('--seeds', type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v3.npz'); hrow, batch, split = T['hist_row'], T['batch'], T['split']
GS = np.load(a.gs).astype(np.float32)                                    # rows = hist_row order
LFf = np.load(os.path.join(a.lf, 'lf_feats_hist.npy')).astype(np.float32); LFi = np.load(os.path.join(a.lf, 'lf_feats_hist_indices.npy'))
assert (hrow[LFi] >= 0).all(); G = GS[hrow[LFi]]; L = LFf; gb = batch[LFi]; gs_ = split[LFi]
def pcaw(X, k):   # PCA-whiten fit on all rows of that modality (unsupervised; no pairing used)
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); P = Vt[:k].T / (S[:k] / np.sqrt(len(X)))
    return lambda Z: (Z - mu) @ P
fG, fL = pcaw(G, a.k), pcaw(L, a.k); Gw, Lw = fG(G), fL(L)
def retr(A, B):
    A = F.normalize(torch.tensor(A), dim=1); B = F.normalize(torch.tensor(B), dim=1); s = A @ B.T; rk = ((s >= s.diag()[:, None]).sum(1) - 1).float()
    return {'R@1': float((rk < 1).float().mean()), 'R@10': float((rk < 10).float().mean()), 'medr': float(rk.median() + 1)}
def mknn(A, B, k=10):
    def nn(X): X = F.normalize(torch.tensor(X), dim=1); s = X @ X.T; s.fill_diagonal_(-9); return s.topk(k, 1).indices
    a_, b_ = nn(A), nn(B); return float(np.mean([len(set(a_[i].tolist()) & set(b_[i].tolist())) / k for i in range(len(A))]))
def siglip_fit(Gt, Lt, dim=128, steps=1500, seed=0):
    torch.manual_seed(seed); Wg = torch.nn.Linear(Gt.shape[1], dim); Wl = torch.nn.Linear(Lt.shape[1], dim); t = torch.nn.Parameter(torch.tensor(np.log(10.))); b = torch.nn.Parameter(torch.tensor(-10.))
    opt = torch.optim.AdamW(list(Wg.parameters()) + list(Wl.parameters()) + [t, b], lr=1e-3, weight_decay=1e-2); Gt, Lt = torch.tensor(Gt), torch.tensor(Lt)
    for _ in range(steps):
        i = torch.randperm(len(Gt))[:min(512, len(Gt))]   # no replacement: duplicate pairs would be labelled negatives; zg = F.normalize(Wg(Gt[i]), dim=1); zl = F.normalize(Wl(Lt[i]), dim=1)
        lg = zg @ zl.T * t.exp() + b; lab = 2 * torch.eye(len(i)) - 1; loss = -F.logsigmoid(lab * lg).mean(); opt.zero_grad(); loss.backward(); opt.step()
    def fg(X):
        with torch.no_grad(): return Wg(torch.tensor(X)).numpy()
    def fl(X):
        with torch.no_grad(): return Wl(torch.tensor(X)).numpy()
    return fg, fl
res = []
protocols = {'slide_b0_to_b1': ((gb == 0), (gb == 1)), 'fov_within': ((gs_ == 0), (gs_ == 2))}
for pname, (trm, tem) in protocols.items():
    te = np.where(tem)[0]; r0 = {'protocol': pname, 'method': 'unaligned_mutual_kNN', 'n': 0, 'mknn_raw': mknn(G[te], L[te]), 'mknn_whitened': mknn(Gw[te], Lw[te])}; res.append(r0); print(r0, flush=True)
    trall = np.where(trm)[0]
    for n in [10, 30, 100, 300, 1000, 3000, len(trall)]:
        for seed in range(a.seeds):
            tr = np.random.default_rng(seed).permutation(trall)[:n]
            U, _, Vt = np.linalg.svd(Gw[tr].T @ Lw[tr]); R = U @ Vt               # orthogonal Procrustes GS->LF
            out = [('procrustes', Gw[te] @ R, Lw[te])]
            if n >= 30:
                out.append(('ridge', Ridge(alpha=10.0).fit(Gw[tr], Lw[tr]).predict(Gw[te]), Lw[te]))
            if n >= 300:
                out.append(('mlp', MLPRegressor((512,), max_iter=300, early_stopping=True, random_state=seed).fit(Gw[tr], Lw[tr]).predict(Gw[te]), Lw[te]))
                fg, fl = siglip_fit(Gw[tr], Lw[tr], seed=seed); out.append(('siglip_linear', fg(Gw[te]), fl(Lw[te])))
            for m, A, B in out:
                r = {'protocol': pname, 'method': m, 'n': int(n), 'seed': seed, 'n_test': int(len(te)), 'chance_R@1': 1 / len(te)}
                r.update({f'GS->LF_{k}': v for k, v in retr(A, B).items()}); r.update({f'LF->GS_{k}': v for k, v in retr(B, A).items()}); res.append(r); print(r, flush=True)
json.dump(res, open(a.out + '/screen.json', 'w'), indent=1)
import csv
keys = sorted({k for r in res for k in r}); w = csv.DictWriter(open(a.out + '/screen.csv', 'w'), fieldnames=keys); w.writeheader(); [w.writerow(r) for r in res]
