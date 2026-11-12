"""N035 POST-HOC (after the frozen verdicts): is the draw-1 HLA-DR real-image readout failure (r -.11) a label or an image problem?
Per single-antibody stain x draw: FOV-grouped 5-fold CV ridge of the curated v5c label on the 6x4 log fluorescence integrals
(fm_prep integ), and the cross-draw transfer of that linear map. Usage: n035_posthoc_hladr.py OUTDIR"""
import sys, json, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, GroupKFold
W = '/hnvme/workspace/v181bb10-bsccm'; F = np.load(W + '/fm_prep.npz', allow_pickle=True); T = np.load(W + '/targets_v5c.npz', allow_pickle=True)
names = [str(n) for n in T['names']]; st = F['stain']; SN = [str(s) for s in F['stain_names']]; I = F['integ']; clean = T['hist_row'] < 0
feats = lambda m: (np.sign(I[m]) * np.log1p(np.abs(I[m]))).reshape(m.sum(), -1); rid = lambda: RidgeCV(alphas=np.logspace(-2, 3, 8)); out = {}
for lab, s in [('single:HLA-DR', 'HLA-DR'), ('single:CD14', 'CD14'), ('single:CD123', 'CD123'), ('single:CD19', 'CD19'), ('single:CD3', 'CD3'), ('single:CD16', 'CD16'), ('single:CD45', 'CD45'), ('single:CD56', 'CD56')]:
    j = names.index(lab); M = {b: clean & T['mask'][:, j] & (st == SN.index(s)) & (T['batch'] == b) for b in (0, 1)}; r = {}
    for b in (0, 1):
        p = cross_val_predict(rid(), feats(M[b]), T['Y'][M[b], j], cv=GroupKFold(5), groups=T['fov'][M[b]]); r[f'within_b{b}'] = float(np.corrcoef(p, T['Y'][M[b], j])[0, 1])
    for a_, b_ in ((0, 1), (1, 0)):
        r[f'cross_b{a_}to{b_}'] = float(np.corrcoef(rid().fit(feats(M[a_]), T['Y'][M[a_], j]).predict(feats(M[b_])), T['Y'][M[b_], j])[0, 1])
    out[lab] = r; print(lab, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(sys.argv[1] + '/n035_posthoc_label_vs_integrals.json', 'w'), indent=1)
