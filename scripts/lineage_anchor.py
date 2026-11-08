"""N008: external lineage anchor. Logistic regression on frozen DinoBloom features of PBC (MGG, CellaVision, 8 classes; 5-fold CV for
in-domain accuracy), applied to BSCCM Wright cells (scale-matched crops). Reports: predicted class composition of BSCCM Wright cells
per batch/slide vs the marker-gated composition of the same blood draw (batch 0 'all' cells: lymph/gran/mono), confidence distribution,
and -- if LF-model predictions for the GS cells are given (--lf-pred npz with G (global idx) and C3 (pred cls3)) -- the agreement matrix
between Wright-morphology class (external, independent of LF) and LF-predicted class.
usage: lineage_anchor.py --out DIR [--lf-pred eval_preds_gs.npz]"""
import argparse, json, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--lf-pred', default=''); ap.add_argument('--size', default='B')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); W = '/hnvme/workspace/v181bb10-bsccm'
X = np.load(f'{W}/emb/pbc_dinobloom_{a.size}.npy'); y = np.load(f'{W}/emb/pbc_labels.npy'); cls = json.load(open(f'{W}/emb/pbc_classes.json'))
G = np.load(f'{W}/emb/giemsa_dinobloom_{a.size}_scalematched.npy'); T = np.load(W + '/targets_v3.npz'); hrow, b, c3 = T['hist_row'], T['batch'], T['cls3']
gidx = np.where(hrow >= 0)[0]; gidx = gidx[np.argsort(hrow[gidx])]          # global index of each histology row
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
cvp = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
res = {'pbc_cv_acc': float((cvp == y).mean()), 'pbc_classes': cls}
clf.fit(X, y); P = clf.predict_proba(G); pred = P.argmax(1); conf = P.max(1)
res['bsccm_conf_pct10_50_90'] = np.percentile(conf, [10, 50, 90]).round(3).tolist()
for bb in (0, 1):
    s = b[gidx] == bb; cnt = np.bincount(pred[s], minlength=len(cls))
    res[f'bsccm_wright_composition_b{bb}'] = {c: round(float(n / s.sum()), 4) for c, n in zip(cls, cnt)}
    hi = s & (conf > 0.8); cnt = np.bincount(pred[hi], minlength=len(cls))
    res[f'bsccm_wright_composition_b{bb}_conf>0.8'] = {c: round(float(n / max(hi.sum(), 1)), 4) for c, n in zip(cls, cnt)} | {'n': int(hi.sum())}
lab = c3 >= 0; res['gated_composition_b0_all_cells'] = {k: round(float((c3[lab] == i).mean()), 4) for i, k in enumerate(['lymphocyte', 'granulocyte', 'monocyte'])}
# coarse mapping of PBC classes to the 3 gated lineages (eosinophil/basophil/ig/neutrophil -> granulocyte)
coarse = {'neutrophil': 1, 'eosinophil': 1, 'basophil': 1, 'ig': 1, 'lymphocyte': 0, 'monocyte': 2}
pc = np.array([coarse.get(cls[k], -1) for k in pred])
res['bsccm_wright_coarse3_b0'] = {k: round(float((pc[b[gidx] == 0] == i).mean()), 4) for i, k in enumerate(['lymphocyte', 'granulocyte', 'monocyte'])} | {'other(platelet/erythroblast)': round(float((pc[b[gidx] == 0] == -1).mean()), 4)}
if a.lf_pred:
    d = np.load(a.lf_pred); m = dict(zip(d['G'].tolist(), d['C3'].tolist())); lf = np.array([m.get(int(g), -9) for g in gidx]); ok = (lf >= 0) & (pc >= 0)
    M = np.zeros((3, 3), int)
    for i, j in zip(pc[ok], lf[ok]): M[i, j] += 1
    res['agreement_wright_vs_LF_coarse3'] = {'rows=Wright(PBC-clf) cols=LF-pred [lym,gran,mono]': M.tolist(), 'agreement': float(np.trace(M) / M.sum()), 'n': int(M.sum())}
np.savez(f'{a.out}/wright_pbc_pred.npz', gidx=gidx, pred=pred, conf=conf, P=P)
json.dump(res, open(f'{a.out}/lineage_anchor.json', 'w'), indent=1); print(json.dumps(res, indent=1))
