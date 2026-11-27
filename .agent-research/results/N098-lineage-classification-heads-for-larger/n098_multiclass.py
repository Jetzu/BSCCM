"""N098 arm 'multiclass' (user 11-27 "train a classifier head on this"; added BEFORE any N098 result was read). One multinomial logistic
head on the frozen n051 2048-d features over 6 lineages (T, B, NK, monocyte, neutrophil, basophil/pDC), trained on the SOURCE draw's
positives of each marker on its own slide (N097 gate rule), class-balanced, C chosen by FOV-grouped 3-fold CV (macro one-vs-rest AUROC).
Test: pairwise AUROC with score logit_A - logit_B on the test draw's positives; slide control = same score, ALL cells of slide A vs ALL
cells of slide B (the classes come from different slides, so a slide shortcut would show up there). Writes n098_multiclass_pairs.csv."""
import glob, itertools, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import n098_lineage_heads as H
B, L, LIN, CS = H.B, H.L, H.LIN, H.CS
rows = []
for d in ('b10', 'b01'):
    z = np.load(f'{B.W}/runs/n051_v10_base_{d}/features.npz'); X, G, ev = z['H'], z['G'], z['is_eval'].astype(bool)
    Xs, Gs, Xt, Gt = X[~ev].astype(np.float32), G[~ev], X[ev].astype(np.float32), G[ev]
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6; Xs = (Xs - mu) / sd; Xt = (Xt - mu) / sd
    idx, cls = [], []
    for k, (_, x) in enumerate(LIN):
        s, p = H.gate(Gs, x); idx.append(s[p]); cls.append(np.full(p.sum(), k))
    idx, cls = np.concatenate(idx), np.concatenate(cls)
    fov = B.SLIDE[Gs[idx]].astype(np.int64) * 100000 + B.FOV[Gs[idx]].astype(np.int64)
    best = None
    for C in CS:
        a = []
        for tr, va in GroupKFold(3).split(idx, cls, fov):
            m = LogisticRegression(C=C, class_weight='balanced', max_iter=300).fit(Xs[idx[tr]], cls[tr])
            a.append(roc_auc_score(cls[va], m.predict_proba(Xs[idx[va]]), multi_class='ovr', average='macro', labels=list(range(len(LIN)))))
        if best is None or np.mean(a) > best[1]:
            best = (C, float(np.mean(a)))
    m = LogisticRegression(C=best[0], class_weight='balanced', max_iter=500).fit(Xs[idx], cls)
    lg = m.decision_function(Xt); print(d, 'C', best, 'n per class', np.bincount(cls).tolist(), flush=True)
    gates = {x: H.gate(Gt, x) for _, x in LIN}
    for (i, (na, a)), (j, (nb, b)) in itertools.combinations(enumerate(LIN), 2):
        sa, pa = gates[a]; sb, pb = gates[b]; sc = lg[:, i] - lg[:, j]
        rows.append(dict(direction=d, A=na, B=nb, arm='multiclass', n_A=int(pa.sum()), n_B=int(pb.sum()),
                         auroc=H.auc(sc[sa[pa]], sc[sb[pb]]), slide_control=H.auc(sc[sa], sc[sb]), C=best[0], cv_macro_auroc_source=best[1]))
df = pd.DataFrame(rows); df.to_csv(f'{HERE}/n098_multiclass_pairs.csv', index=False)
print(df[['direction', 'A', 'B', 'auroc', 'slide_control']].round(3).to_string(index=False))
