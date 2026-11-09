"""Re-score saved eval predictions (runs/<run>/eval_preds.npz: P, G) of LF->MK runs against other targets files (N023 metric-only control).
Protocol and training rows are rebuilt from the run's own result.json args (protocol, include_gs, max_train) with the same rules as
train_lf.py -- never guessed from the run name (red-team F6). Reports per-target r, the mean over all targets with test labels
(r_mean_avail), the mean over targets with >= 1000 labels in the run's training rows under that targets file (r_mean_valid, same rule as
train_lf.py r_mean_markers_valid), and the mean over a FIXED common target set (--common, default: the 11 targets excluding
grp:CD123/HLA-DR/CD14 = the frozen 'LF->MK common-11' benchmark of N023)."""
import argparse, json, numpy as np
from bsccm.data import W
COMMON11 = ['single:CD45', 'single:CD123', 'single:CD19', 'single:CD56', 'single:CD14', 'single:CD16', 'single:HLA-DR', 'single:CD3',
            'grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16']
ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--targets', nargs='+', default=['targets_v4.npz', 'targets_v5.npz'])
ap.add_argument('--common', nargs='+', default=COMMON11); ap.add_argument('--out', required=True); a = ap.parse_args()
TT = {t: np.load(W + '/' + t, allow_pickle=True) for t in a.targets}; rows = []
def train_rows(T, args):
    Mk, c3, sp, b = T['mask'], T['cls3'], T['split'], T['batch']
    has = Mk.any(1) | (c3 >= 0)
    if not int(args.get('include_gs', 0)): has &= T['hist_row'] < 0
    p = args['protocol']
    tr = {'fov': (sp == 0), 'b1to0': (b == 1), 'b0to1': (b == 0)}[p] & has
    tr = np.where(tr)[0]
    if int(args.get('max_train', 0)): tr = np.random.default_rng(0).permutation(tr)[:int(args['max_train'])]
    return tr
for run in a.runs:
    args = json.load(open(f'{W}/runs/{run}/result.json'))['args']; proto = args['protocol']
    E = np.load(f'{W}/runs/{run}/eval_preds.npz'); P, G = E['P'], E['G']
    for t, T in TT.items():
        Y, M, names = T['Y'], T['mask'], [str(n) for n in T['names']]
        trained = M[train_rows(T, args)].sum(0) >= 1000
        rs = {}
        for j, n in enumerate(names[:12]):
            v = M[G, j]
            if v.sum() > 20: rs[n] = float(np.corrcoef(P[v, j], Y[G[v], j])[0, 1])
        valid = [rs[n] for j, n in enumerate(names[:12]) if n in rs and trained[j]]
        missing = [n for n in a.common if n not in rs]
        common = float(np.mean([rs[n] for n in a.common])) if not missing else float('nan')
        rows.append(dict(run=run, protocol=proto, targets=t, r_mean_avail=float(np.mean(list(rs.values()))), n_avail=len(rs), r_mean_valid=float(np.mean(valid)),
                         n_valid=len(valid), r_mean_common=common, common_missing=missing, **{f'r/{k}': v for k, v in rs.items()}))
        print(f'{run:22s} {proto:5s} {t:15s} avail(n={len(rs)}) {rows[-1]["r_mean_avail"]:.4f}  valid(n={len(valid)}) {rows[-1]["r_mean_valid"]:.4f}  '
              f'common{len(a.common)} {common:.4f}  A={rs.get("grp:CD123/HLA-DR/CD14", float("nan")):.3f}')
json.dump(rows, open(a.out, 'w'), indent=1)
