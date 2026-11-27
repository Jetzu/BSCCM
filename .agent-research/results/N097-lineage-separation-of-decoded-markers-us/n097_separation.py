"""N097 part 2 (user 11-27: "one image that shows how well all lines can be separated" + "correlation plots with labelled Pearson r
for a few important cell line defining markers"). Saved predictions only.
Pairwise lineage separability: for lineages A and B with defining markers a and b, every cell gets the score
z(pred a) - z(pred b) (z over all test cells); AUROC(A vs B) = probability that a random A cell scores higher than a random B cell
(.5 = coin flip, 1 = perfect). No classifier is fitted.
BSCCM: lineage = positives of its marker on its own single-stain slide of the TEST draw (same gates as n097_lineage.py);
populations of a pair come from different slides -> slide control: the same score on ALL cells of slide a vs ALL cells of slide b
(every slide samples the same blood draw, so ~.5 unless predictions carry slide-specific offsets).
Sandbox: lineages = hierarchical gates on the held-out donor's own measured labels; also the measured markers as reference."""
import glob, os, sys, itertools
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); R = '/home/hpc/v181bb/v181bb10/BSCCM'
sys.path.insert(0, HERE); import n097_lineage as L
B, SE = L.B, L.SE
BS_LIN = [('T', 'CD3'), ('B', 'CD19'), ('NK', 'CD56'), ('monocyte', 'CD14'), ('neutrophil', 'CD16'), ('basophil/pDC', 'CD123')]
SB_LIN = [('CD4 T', 'T4', 'CD4'), ('CD8 T', 'T8', 'CD8'), ('B', 'B', 'CD19'), ('NK', 'NK', 'CD56'), ('monocyte', 'mono', 'CD14'),
          ('neutrophil', 'neut', 'CD15'), ('eosinophil', 'eos', 'Siglec8')]


def auc(pos_score, neg_score):
    return roc_auc_score(np.r_[np.ones(len(pos_score)), np.zeros(len(neg_score))], np.r_[pos_score, neg_score])


def bsccm():
    rows, sc = [], []
    for d in ('b10', 'b01'):
        m, G = B.members(d); P = np.mean([m[k] for k in m], 0); Z = (P - P.mean(0)) / P.std(0)
        pos, neg = {}, {}
        for x in L.LIN:
            s = np.where(B.M[G, L.col[x]])[0]; y = B.Y[G[s], L.col[x]]; t = L.valley(y)
            if x not in ('CD16', 'CD45'):
                med = np.median(y); t = max(t, med + 3 * 1.4826 * np.median(np.abs(y - med)))
            pos[x], neg[x] = s[y > t], s[y <= t]
            sc.append(pd.DataFrame(dict(direction=d, marker=x, measured=y, predicted=P[s, L.col[x]])))
        for (na, a), (nb, b) in itertools.permutations(BS_LIN, 2):
            score = Z[:, L.col[a]] - Z[:, L.col[b]]
            rows.append(dict(direction=d, A=na, B=nb, n_A=len(pos[a]), n_B=len(pos[b]), auroc=auc(score[pos[a]], score[pos[b]]),
                             slide_control=auc(score[np.r_[pos[a], neg[a]]], score[np.r_[pos[b], neg[b]]])))   # ALL cells of slide a vs slide b
    return pd.DataFrame(rows), pd.concat(sc)


def sandbox():
    lab = np.load(L.SB + '/external/sbiad452_pack/labels.npz', allow_pickle=False); raw = lab['I_raw'].astype(np.float64); MK = list(SE.M)
    rows = []
    for d in (0, 1, 2):
        for arm, r in (('all markers per cell', 'n080_full'), ('one marker per cell', 'n080_single')):
            z = np.load(f'{L.SB}/runs/{r}_d{d}/eval_preds.npz', allow_pickle=True); P, Yt, ix = z['P'], z['Y'], z['rows']
            t = SE.gate(raw[ix])
            sets = [(arm, P)] + ([('measured', Yt)] if arm == 'all markers per cell' else [])
            for a_, Q in sets:
                Z = (Q - Q.mean(0)) / Q.std(0)
                for (na, ta, ma), (nb, tb, mb) in itertools.permutations(SB_LIN, 2):
                    s = Z[:, MK.index(ma)] - Z[:, MK.index(mb)]
                    rows.append(dict(donor=d, arm=a_, A=na, B=nb, n_A=int((t == ta).sum()), n_B=int((t == tb).sum()),
                                     auroc=auc(s[t == ta], s[t == tb])))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    b, sc = bsccm(); b.to_csv(f'{HERE}/sep_bsccm.csv', index=False); sc.to_csv(f'{HERE}/scatter_bsccm.csv', index=False)
    s = sandbox(); s.to_csv(f'{HERE}/sep_sandbox.csv', index=False)
    print(b.groupby(['A', 'B'])[['auroc', 'slide_control']].mean().round(3).to_string())
    print('slide control mean |AUROC-.5|:', round(float((b.slide_control - .5).abs().mean()), 3), 'max', round(float((b.slide_control - .5).abs().max()), 3))
    print(s.groupby(['arm', 'A', 'B']).auroc.mean().unstack().round(2).to_string())
    print(sc.groupby(['direction', 'marker']).apply(lambda g: B.pr(g.measured.values, g.predicted.values)).round(3).to_string())
