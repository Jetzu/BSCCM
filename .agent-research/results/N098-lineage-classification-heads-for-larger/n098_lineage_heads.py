"""N098 (user 11-27: "make larger separations, CD3 and CD19 are not that well separable"). Frozen n051_v10_base features (2048-d),
both directions, CPU. PRE-REGISTERED in the N098 item before results.
Arms (same test cells, same test-draw lineage gates as N097 fig14):
  regression  : score = z(regression output of marker a) - z(output of marker b)  (n051's own linear head; = fig14 method)
  heads       : per lineage marker a logistic positive-vs-negative head on the 2048-d features, trained ONLY on cells of marker a's
                own single-stain slide in the SOURCE draw (positives by the N097 gate rule on source labels, class-balanced, C chosen by
                FOV-grouped 3-fold CV on source rows) -> score = z(logit a) - z(logit b)
Metric: pairwise AUROC between lineages (positives of each marker on its own test-draw slide). Slide control: the same score on ALL
cells of slide a vs ALL cells of slide b (~.5 expected). Writes n098_pairs.csv, n098_heads.csv."""
import glob, itertools, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); R = '/home/hpc/v181bb/v181bb10/BSCCM'
sys.path.insert(0, glob.glob(R + '/.agent-research/results/N097-*')[0]); import n097_lineage as L
B = L.B; W = B.W
LIN = [('T', 'CD3'), ('B', 'CD19'), ('NK', 'CD56'), ('monocyte', 'CD14'), ('neutrophil', 'CD16'), ('basophil/pDC', 'CD123')]
CS = (1e-3, 1e-2, 1e-1)


def gate(G, x):
    s = np.where(B.M[G, L.col[x]])[0]; y = B.Y[G[s], L.col[x]]; t = L.valley(y)
    if x not in ('CD16', 'CD45'):
        med = np.median(y); t = max(t, med + 3 * 1.4826 * np.median(np.abs(y - med)))
    return s, y > t


def auc(p, n):
    return roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])


def main():
    pairs, heads = [], []
    for d in ('b10', 'b01'):
        z = np.load(f'{W}/runs/n051_v10_base_{d}/features.npz'); H, G, ev, P = z['H'], z['G'], z['is_eval'].astype(bool), z['P']
        Hs, Gs, Ht, Gt, Pt = H[~ev].astype(np.float32), G[~ev], H[ev].astype(np.float32), G[ev], P[ev]
        mu, sd = Hs.mean(0), Hs.std(0) + 1e-6; Hs = (Hs - mu) / sd; Ht = (Ht - mu) / sd
        logit = {}
        for _, x in LIN:
            s, ypos = gate(Gs, x)
            fov = B.SLIDE[Gs[s]].astype(np.int64) * 100000 + B.FOV[Gs[s]].astype(np.int64)
            best = None
            for C in CS:   # FOV-grouped 3-fold CV on the source slide only
                a = []
                for tr, va in GroupKFold(3).split(s, ypos, fov):
                    m = LogisticRegression(C=C, class_weight='balanced', max_iter=300).fit(Hs[s[tr]], ypos[tr])
                    a.append(roc_auc_score(ypos[va], m.decision_function(Hs[s[va]])))
                if best is None or np.mean(a) > best[1]:
                    best = (C, float(np.mean(a)))
            m = LogisticRegression(C=best[0], class_weight='balanced', max_iter=500).fit(Hs[s], ypos)
            logit[x] = m.decision_function(Ht)
            heads.append(dict(direction=d, marker=x, n_source=len(s), frac_pos_source=float(ypos.mean()), C=best[0], cv_auroc_source=best[1]))
            print(d, x, heads[-1], flush=True)
        Zr = (Pt - Pt.mean(0)) / Pt.std(0)
        Zh = {x: (v - v.mean()) / v.std() for x, v in logit.items()}
        gates = {x: gate(Gt, x) for _, x in LIN}
        for (na, a), (nb, b) in itertools.combinations(LIN, 2):
            sa, pa = gates[a]; sb, pb = gates[b]
            for arm, score in (('regression', Zr[:, L.col[a]] - Zr[:, L.col[b]]), ('heads', Zh[a] - Zh[b])):
                pairs.append(dict(direction=d, A=na, B=nb, arm=arm, n_A=int(pa.sum()), n_B=int(pb.sum()),
                                  auroc=auc(score[sa[pa]], score[sb[pb]]), slide_control=auc(score[sa], score[sb])))
    P_ = pd.DataFrame(pairs); P_.to_csv(f'{HERE}/n098_pairs.csv', index=False); pd.DataFrame(heads).to_csv(f'{HERE}/n098_heads.csv', index=False)
    print(P_.pivot_table(index=['A', 'B'], columns=['arm', 'direction'], values='auroc').round(3).to_string())
    print(P_.groupby(['arm', 'direction'])[['auroc', 'slide_control']].agg(['mean', 'min', 'max']).round(3).to_string())


if __name__ == '__main__':
    main()
