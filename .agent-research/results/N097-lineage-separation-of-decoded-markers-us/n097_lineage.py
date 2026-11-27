"""N097 (user 11-27: "CD3 and CD19 must be anticorrelated - are they, can the models separate cell lineages?").
Saved predictions only (no training). Writes CSV/NPZ next to this file; figures: figures/overview/make_lineage.py.
BSCCM: CD3 and CD19 are never measured on the same cell (single-stain slides). Lineage test = true positives of each lineage
marker on its own slide (2-component GMM valley of the test draw's label, and for the rare-positive markers also > median + 3 MAD; evaluation only) -> what the model predicts for ALL
markers on those cells (record ensemble E8 + members, both directions).
Sandbox (S-BIAD452 ImageStream, 9 markers on every cell): gated true types (N080 hierarchical gates on the test donor's own
labels) -> true vs predicted CD3-CD19 correlation in lymphocytes, T-vs-B separation, type x marker profiles."""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__)); R = '/home/hpc/v181bb/v181bb10/BSCCM'
sys.path.insert(0, glob.glob(R + '/.agent-research/results/N096-*')[0]); import n096_bsccm as B   # members(), names, Y, M (no side effects)
sys.path.insert(0, glob.glob(R + '/.agent-research/results/N087-*')[0] + '/codex'); import n087_sandbox_eval as SE   # gate(), valley()
W = B.W; names, Y, M = B.names, B.Y, B.M
LIN = ['CD3', 'CD19', 'CD56', 'CD14', 'CD16', 'CD123', 'HLA-DR', 'CD45']
col = {m: names.index('single:' + m) for m in LIN}


def valley(x):
    g = GaussianMixture(2, random_state=0).fit(x[:, None]); mu = np.sort(g.means_.ravel())
    grid = np.linspace(mu[0], mu[1], 400); return float(grid[np.argmin(np.exp(g.score_samples(grid[:, None])))])


def zs(P):
    return (P - P.mean(0)) / P.std(0)


def bsccm():
    prof, auc, corr, cells = [], [], [], []
    for d in ('b10', 'b01'):
        m, G = B.members(d); m = {'E8 record': np.mean([m[k] for k in m], 0), **m}
        pos = {}
        for x in LIN:
            s = np.where(M[G, col[x]])[0]; y = Y[G[s], col[x]]; t = valley(y)
            if x not in ('CD16', 'CD45'):   # rare-positive lineage markers: the valley can fall inside a unimodal bulk (CD19 draw 1 -> 60 %
                med = np.median(y); t = max(t, med + 3 * 1.4826 * np.median(np.abs(y - med)))   # positive); require > median + 3 MAD too
            pos[x] = s[y > t]; neg = s[y <= t]
            prof.append(dict(direction=d, model='threshold', lineage=x, n_stained=len(s), n_pos=len(pos[x]), frac_pos=len(pos[x]) / len(s), thr=t))
            pos[x + '-'] = neg
        for k, P in m.items():
            Z = zs(P)
            for x in LIN:   # mean predicted z of every marker on the true positives of lineage x
                for yv in LIN:
                    prof.append(dict(direction=d, model=k, lineage=x, marker=yv, mean_pred_z=float(Z[pos[x], col[yv]].mean()), n=len(pos[x])))
                auc.append(dict(direction=d, model=k, test=f'{x}+ vs {x}- (own slide)',
                                auroc=roc_auc_score(np.r_[np.ones(len(pos[x])), np.zeros(len(pos[x + '-']))], P[np.r_[pos[x], pos[x + '-']], col[x]])))
            T_, B_ = pos['CD3'], pos['CD19']; lab = np.r_[np.ones(len(B_)), np.zeros(len(T_))]; rows = np.r_[B_, T_]
            auc.append(dict(direction=d, model=k, test='B vs T by predicted CD19', auroc=roc_auc_score(lab, P[rows, col['CD19']])))
            auc.append(dict(direction=d, model=k, test='T vs B by predicted CD3', auroc=roc_auc_score(1 - lab, P[rows, col['CD3']])))
            auc.append(dict(direction=d, model=k, test='B vs T by predicted CD19 - CD3', auroc=roc_auc_score(lab, Z[rows, col['CD19']] - Z[rows, col['CD3']])))
            lym = np.where(B.cls3[G] == 0)[0]   # gated lymphocytes (draw-0 'all' slide only -> b10 test)
            corr.append(dict(direction=d, model=k, cells='all test cells', n=len(G), r=B.pr(P[:, col['CD3']], P[:, col['CD19']])))
            if len(lym) > 100:
                corr.append(dict(direction=d, model=k, cells='gated lymphocytes', n=len(lym), r=B.pr(P[lym, col['CD3']], P[lym, col['CD19']])))
            corr.append(dict(direction=d, model=k, cells='true T + B cells', n=len(rows), r=B.pr(P[rows, col['CD3']], P[rows, col['CD19']])))
            if k == 'E8 record':
                for grp, rr in (('T (CD3+ slide)', T_), ('B (CD19+ slide)', B_), ('CD3- (CD3 slide)', pos['CD3-']), ('CD19- (CD19 slide)', pos['CD19-'])):
                    cells.append(pd.DataFrame(dict(direction=d, group=grp, pred_CD3=Z[rr, col['CD3']], pred_CD19=Z[rr, col['CD19']])))
    return pd.DataFrame(prof), pd.DataFrame(auc), pd.DataFrame(corr), pd.concat(cells)


SB = '/hnvme/workspace/v181bb10-bsccm'


def sandbox():
    lab = np.load(SB + '/external/sbiad452_pack/labels.npz', allow_pickle=False); raw = lab['I_raw'].astype(np.float64)
    MK = list(SE.M); i3, i19 = MK.index('CD3'), MK.index('CD19')
    runs = {'all markers per cell': 'n080_full', 'one marker per cell': 'n080_single'}
    for lam in (1, 10, 100):
        runs[f'one marker + corr loss {lam}'] = f'n096/sb_single_corr{lam}'
    runs['all markers + corr loss 10'] = 'n096/sb_full_corr10'
    rows, prof, cells = [], [], []
    for d in (0, 1, 2):
        truth_done = False
        for arm, r in runs.items():
            f = f'{SB}/runs/{r}_d{d}/eval_preds.npz'
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True); P, Yt, ix = z['P'], z['Y'], z['rows']
            t = SE.gate(raw[ix]); lym = np.isin(t, SE.LYM); Tm = np.isin(t, ('T4', 'T8', 'Tother')); Bm = t == 'B'
            sets = [('truth', Yt)] if not truth_done else []
            for a, Q in sets + [(arm, P)]:
                rows.append(dict(donor=d, arm=a, r_CD3_CD19_lymph=B.pr(Q[lym, i3], Q[lym, i19]), n_lymph=int(lym.sum()),
                                 auroc_B_vs_T=roc_auc_score(np.r_[np.ones(Bm.sum()), np.zeros(Tm.sum())], np.r_[Q[Bm, i19] - Q[Bm, i3], Q[Tm, i19] - Q[Tm, i3]]),
                                 n_T=int(Tm.sum()), n_B=int(Bm.sum()), r_CD3_vs_measured=B.pr(Q[:, i3], Yt[:, i3]), r_CD19_vs_measured=B.pr(Q[:, i19], Yt[:, i19])))
                Zq = (Q - Q.mean(0)) / Q.std(0)
                for ty in ('Tother', 'T4', 'T8', 'B', 'NK', 'mono', 'neut', 'eos'):
                    k = t == ty
                    if k.sum() >= 20:
                        prof += [dict(donor=d, arm=a, type=ty, marker=mk, mean_z=float(Zq[k, j].mean()), n=int(k.sum())) for j, mk in enumerate(MK)]
                if d == 0 and a in ('truth', 'all markers per cell', 'one marker per cell'):
                    cells.append(pd.DataFrame(dict(donor=d, arm=a, type=t[lym], CD3=Zq[lym, i3], CD19=Zq[lym, i19])))
            truth_done = True
    return pd.DataFrame(rows), pd.DataFrame(prof), pd.concat(cells)


if __name__ == '__main__':
    p, a, c, cl = bsccm()
    p.to_csv(f'{HERE}/bsccm_lineage_profiles.csv', index=False); a.to_csv(f'{HERE}/bsccm_auroc.csv', index=False)
    c.to_csv(f'{HERE}/bsccm_cd3_cd19_corr.csv', index=False); cl.to_csv(f'{HERE}/bsccm_tb_cells.csv', index=False)
    print(p[p.model == 'threshold'].round(3).to_string(index=False))
    print(a[a.model == 'E8 record'].round(3).to_string(index=False)); print(c[c.model == 'E8 record'].round(3).to_string(index=False))
    s, sp, sc = sandbox()
    s.to_csv(f'{HERE}/sandbox_cd3_cd19.csv', index=False); sp.to_csv(f'{HERE}/sandbox_type_profiles.csv', index=False); sc.to_csv(f'{HERE}/sandbox_lymph_cells.csv', index=False)
    print(s.round(3).to_string(index=False))
