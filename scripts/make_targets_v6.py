"""Targets v6 (N025) = targets_v5 with autofl kept ONLY on unstained slides (labels only). Stained-slide autofl carries co-stain
contamination: LF-class relations of autofl reverse between co-stained slides (N023 red-team F4), and v4 vs v5 teachers' autofl outputs on the
same Giemsa cells correlate -.44 (N025 diagnosis). Unstained autofl is the Giemsa-chain truth. Marker targets unchanged (== v5)."""
import os as _os; CLEAN = _os.environ.get('BSCCM_CLEAN', '0') == '1'; S_ = 'c' if CLEAN else ''   # N029: CLEAN=1 -> no Giemsa (hist_row>=0) cell enters ANY fit/calibration; outputs targets_v*c
import numpy as np
W = '/hnvme/workspace/v181bb10-bsccm'
T = dict(np.load(W + f'/targets_v5{S_}.npz', allow_pickle=True)); T['slide_names'] = T['slide_names'].astype(str)
names = [str(n) for n in T['names']]; j = names.index('autofl'); Y = T['Y'].copy(); Mk = T['mask'].copy(); sl, b, sp = T['slide'], T['batch'], T['split']
uns = np.char.startswith(T['slide_names'], 'antibodies_unstained/')[sl]
drop = Mk[:, j] & ~uns; Mk[drop, j] = False; Y[drop, j] = 0.0
for bb in (0, 1):   # re-standardise within batch over the remaining (unstained) cells, stats on split-0 rows (v3 rule)
    sel = Mk[:, j] & (b == bb); ref = sel & (sp == 0) & ((T['hist_row'] < 0) if CLEAN else True); Y[sel, j] = (Y[sel, j] - Y[ref, j].mean()) / Y[ref, j].std()
print(f'autofl kept on {Mk[:, j].sum()} unstained cells ({(Mk[:, j] & (T["hist_row"] < 0)).sum()} non-GS), dropped {drop.sum()} stained-slide cells')
T['Y_v5'] = T['Y']; T['Y'] = Y.astype(np.float32); T['mask'] = Mk
np.savez(W + f'/targets_v6{S_}.npz', **T); print(f'wrote targets_v6{S_}.npz')
