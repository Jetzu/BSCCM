"""N067: consensus TRAINING targets. For every GOAL target and each draw separately (so a direction's training targets use only its
training draw): z(v10) and z(local-corner-background reconstruction, runs/n067_alllabels/half_labels.npz 'full') standardised on that
draw's split-0 cells with both labels; consensus = mean of the two z's, rescaled to v10's split-0 mean/sd of that draw.
PRE-REGISTERED label-only rule (no model, no held-out labels): use the consensus for (draw, target) only if r(v10, local) >= .5 on that
draw's split-0 cells, else keep v10; cells without a finite local label keep v10. Evaluation stays on targets_v5cx_v10.npz (scorer).
writes /hnvme/workspace/v181bb10-bsccm/targets_v5cx_v10cons.npz and n067_rule.csv; ALSO (for N075, label separation)
targets_v5cx_localbg.npz = the local-background version alone on the v10 scale of each draw (same rescaling, no rule; mask = v10 mask AND
finite local label), used only as an auxiliary training head target."""
import numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = dict(np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True))
H = np.load(W + '/runs/n067_alllabels/half_labels.npz'); hn = [str(n) for n in H['names']]
names = [str(n) for n in T['names']]; Y = T['Y'].copy(); M = T['mask']; B = T['batch']; S = T['split']
goal = [j for j, n in enumerate(names) if n.startswith('single:') or (n.startswith('grp:') and n != 'grp:CD123/HLA-DR/CD14')]
loc = np.full((len(Y), len(names)), np.nan); loc[H['G']] = np.stack([H['full'][:, hn.index(names[j])] if names[j] in hn else np.full(len(H['G']), np.nan) for j in range(len(names))], 1)
rows = []
for j in goal:
    for b in (0, 1):
        d = (B == b) & M[:, j]; both = d & np.isfinite(loc[:, j]); cal = both & (S == 0)
        y, l = T['Y'][cal, j], loc[cal, j]; r = float(np.corrcoef(y, l)[0, 1]); use = r >= .5
        if use:
            zy = (T['Y'][both, j] - y.mean()) / y.std(); zl = (loc[both, j] - l.mean()) / l.std(); c = (zy + zl) / 2
            c0 = ((T['Y'][cal, j] - y.mean()) / y.std() + (l - l.mean()) / l.std()) / 2
            Y[both, j] = y.mean() + y.std() * (c - c0.mean()) / c0.std()
        rows.append(dict(target=names[j], draw=b, n_split0=int(cal.sum()), n_masked=int(d.sum()), n_local=int(both.sum()), r_v10_local_split0=r,
                         consensus=bool(use), r_new_vs_v10_all=float(np.corrcoef(Y[d, j], T['Y'][d, j])[0, 1])))
df = pd.DataFrame(rows); print(df.round(3).to_string(index=False)); df.to_csv('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N067-consensus-background-training-targets-tr/n067_rule.csv', index=False)
YL = np.zeros_like(T['Y']); ML = np.zeros_like(M)
for j in goal:
    for b in (0, 1):
        d = (B == b) & M[:, j]; both = d & np.isfinite(loc[:, j]); cal = both & (S == 0)
        y, l = T['Y'][cal, j], loc[cal, j]
        YL[both, j] = y.mean() + y.std() * (loc[both, j] - l.mean()) / l.std(); ML[both, j] = True
TL = dict(T); TL['Y'] = YL.astype(T['Y'].dtype); TL['mask'] = ML; np.savez(W + '/targets_v5cx_localbg.npz', **TL); print('wrote targets_v5cx_localbg.npz', int(ML.sum()))
T['Y'] = Y; np.savez(W + '/targets_v5cx_v10cons.npz', **T); print('wrote targets_v5cx_v10cons.npz')
