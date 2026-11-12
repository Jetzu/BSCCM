"""N038 assumption breaker H: what is the image-predictable HLA-DR label variance inside draw-1 lymphocytes (FOV models r .71, draw-0-trained
ensemble .10; draw 0: all models ~.0-.1)? Within predicted lymphocytes (C3 of the v5 r18 FOV model, split-2 cells) of each draw's HLA-DR
slide: r(label, authors' autofl unmixed), r(label, each raw channel shading_corrected), 2-GMM positive fraction; and r(FOV-model prediction,
autofl unmixed)."""
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt = T['Y'], T['mask'], T['batch']; sm = pd.read_parquet(W + '/bsccm_sm.parquet')
af = np.arcsinh(sm['autofluor_single_antibody_model_unmixed'].values / 10); CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G'], E['C3']
Pf, Gf, C3f = load('n024_r18_v5_fov'); pf = pd.DataFrame(Pf, index=Gf); c3 = pd.Series(C3f, index=Gf)
for m in ['HLA-DR', 'CD19', 'CD14']:
    j = names.index(f'single:{m}')
    for b in (0, 1):
        g = Gf[(bt[Gf] == b) & M[Gf, j]]; k = c3.loc[g].values; y = Y[g, j]; p = pf.loc[g, j].values
        for kk, kn in ((0, 'lym'), (2, 'mono')):
            s = k == kk; ys = y[s]
            gm = GaussianMixture(2, random_state=0, n_init=3).fit(ys[:, None]); hi = np.argmax(gm.means_[:, 0])
            chs = {c: round(float(np.corrcoef(ys, np.arcsinh(sm[f'Fluor_{c}_shading_corrected'].values[g][s] / 10))[0, 1]), 2) for c in CH}
            print(f'{m:6s} draw {b} {kn}: n {s.sum():4d} r(label, autofl_unmixed) {np.corrcoef(ys, af[g][s])[0,1]:+.2f}  r(FOVpred, autofl) {np.corrcoef(p[s], af[g][s])[0,1]:+.2f}  '
                  f'r(FOVpred,label) {np.corrcoef(p[s], ys)[0,1]:+.2f}  GMM hi frac {gm.weights_[hi]:.2f}  r(label, raw ch) {chs}')
