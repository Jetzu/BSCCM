"""N010: simulated mosaic with ground truth (validation of the transitive GS->LF->MK claim and its error bounds).
Inside the label-free data, two channel subsets play the roles of the two image modalities:
  A = pseudo-GS (e.g. brightfield + phase), B = pseudo-pivot (e.g. darkfield subset), M = measured markers.
Disjoint cell sets (by FOV split + random halves of the training FOVs): S1 = A-B pairs only (markers hidden), S2 = B-M pairs only
(A hidden), S3 = test cells where A AND M are known (truth). Linear (ridge) models on frozen features:
  teacher  T: B -> M on S2;   student D (distillation): A -> T(B) on S1;   oracle O: A -> M on S2 (uses the A-M pairing; ceiling)
Reported on S3 per marker: true RMSE/r of D vs M (the quantity we can never measure for real GS), a = RMSE(T(B) vs M) on S3-like held-out
B-M cells, b = RMSE(D(A) vs T(B)) on held-out A-B cells, the bound interval [(a-b)^2, (a+b)^2] for the true MSE and the CIA value a^2+b^2,
and whether the true MSE falls inside. Pivot strength is varied (B = 1..17 darkfield channels): weaker pivot = more A-information about M
that B lacks = stronger CIA violation.
usage: sim_mosaic.py --feats DIR (feats_<subset>.npy per subset, rows = global index of cells listed in rows.npy) --out DIR"""
import argparse, json, os, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--feats', required=True); ap.add_argument('--out', required=True); ap.add_argument('--A', default='A_bf_phase')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v4.npz'); names = list(T['names'])
rows = np.load(os.path.join(a.feats, 'rows.npy')); sp = T['split'][rows]; Y = T['Y'][rows]; Mk = T['mask'][rows]
rng = np.random.default_rng(0); tr = np.where(sp == 0)[0]; half = rng.random(len(tr)) < 0.5
S1, S2 = tr[half], tr[~half]; S3 = np.where(sp == 2)[0]; V = np.where(sp == 1)[0]          # V: held-out cells for measuring a and b
def ridge(X, y, alpha=10.0):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym)); return lambda Q: ((Q - mu) / sd) @ w + ym
def pr(p, y): p, y = p - p.mean(), y - y.mean(); return float((p * y).sum() / np.sqrt((p * p).sum() * (y * y).sum() + 1e-12))
FA = np.load(os.path.join(a.feats, f'feats_{a.A}.npy')).astype(np.float64)
res = []
for fb in sorted(f for f in os.listdir(a.feats) if f.startswith('feats_B')):
    FB = np.load(os.path.join(a.feats, fb)).astype(np.float64); bname = fb[6:-4]
    for j, n in enumerate(names[:12]):
        s2, s3, v = S2[Mk[S2, j]], S3[Mk[S3, j]], V[Mk[V, j]]
        if len(s2) < 200 or len(s3) < 100: continue
        Tm = ridge(FB[s2], Y[s2, j]); Dm = ridge(FA[S1], Tm(FB[S1])); Om = ridge(FA[s2], Y[s2, j])
        true_mse = float(np.mean((Dm(FA[s3]) - Y[s3, j]) ** 2))
        a_ = float(np.sqrt(np.mean((Tm(FB[v]) - Y[v, j]) ** 2)))           # measurable: teacher error on held-out B-M cells
        b_ = float(np.sqrt(np.mean((Dm(FA[V]) - Tm(FB[V])) ** 2)))          # measurable: student-vs-teacher on held-out A-B cells
        lo, hi = (a_ - b_) ** 2, (a_ + b_) ** 2
        res.append(dict(pivot=bname, marker=n, n_test=int(len(s3)), true_r_distilled=pr(Dm(FA[s3]), Y[s3, j]), true_r_teacher=pr(Tm(FB[s3]), Y[s3, j]),
                        true_r_oracle_direct=pr(Om(FA[s3]), Y[s3, j]), true_mse=true_mse, a=a_, b=b_, bound_lo=lo, bound_hi=hi, cia_mse=a_ ** 2 + b_ ** 2,
                        inside=bool(lo <= true_mse <= hi), cia_rel_err=float((a_ ** 2 + b_ ** 2 - true_mse) / true_mse)))
        print(json.dumps(res[-1]), flush=True)
json.dump(res, open(a.out + '/sim_mosaic.json', 'w'), indent=1)
import csv
w = csv.DictWriter(open(a.out + '/sim_mosaic.csv', 'w'), fieldnames=list(res[0])); w.writeheader(); [w.writerow(r) for r in res]
print('coverage of no-assumption bound:', np.mean([r['inside'] for r in res]), '| mean CIA rel err:', np.mean([r['cia_rel_err'] for r in res]))
