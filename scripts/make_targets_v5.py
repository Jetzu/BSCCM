"""Targets v5 (N023) = targets_v4 + label QC (labels only; LF inputs untouched).
 1. MASK grp:CD123/HLA-DR/CD14 on batch-1 'all' slides: the dye is 3-5x dimmer than in batch 0 and shares the 426-446 channel with the
    much brighter CD45/CD16 dyes, so the unmixed value is residual; the fluorescence-independent LF-class contrast mono>lym is +1.21 SD in
    batch 0 but -0.08 / -0.28 on the two batch-1 slides (scripts/label_qc.py).
 2. PER-SLIDE robust normalisation where a batch has several slides for one target (batch-1 'all' rep0/rep1 for grp:*, and autofl on every
    slide): slides of one batch come from the same blood draw, so slide offsets/scales are technical (rep0 is ~5x dimmer; autofl shifts up to
    2 SD with the co-stained dye). y <- (y - median_slide) / (IQR_slide / 1.349), then z-scored within (batch, target-group) as before.
    Single-antibody targets have one slide per batch (already per-slide z) -> unchanged.
Keeps Y_v4 (and Y_v3); adds qc_masked (bool rows x targets)."""
import os as _os; CLEAN = _os.environ.get('BSCCM_CLEAN', '0') == '1'; S_ = 'c' if CLEAN else ''   # N029: CLEAN=1 -> no Giemsa (hist_row>=0) cell enters ANY fit/calibration; outputs targets_v*c
import numpy as np
W = '/hnvme/workspace/v181bb10-bsccm'
T = dict(np.load(W + f'/targets_v4{S_}.npz', allow_pickle=True)); T['slide_names'] = T['slide_names'].astype(str)
names = [str(n) for n in T['names']]; Y = T['Y'].astype(np.float64).copy(); Mk = T['mask'].copy(); sl, b, sn, sp = T['slide'], T['batch'], T['slide_names'], T['split']
is_all = np.char.startswith(sn.astype(str), 'antibodies_all/')[sl]
qc = np.zeros_like(Mk); OKC = (T['hist_row'] < 0) if CLEAN else np.ones(len(sp), bool)
jA = names.index('grp:CD123/HLA-DR/CD14')
rA = is_all & (b == 1) & Mk[:, jA]; qc[rA, jA] = True; Mk[rA, jA] = False
print(f'masked grp:CD123/HLA-DR/CD14 on batch-1 all: {rA.sum()} cells')
for j, n in enumerate(names):
    if not (n.startswith('grp:') or n == 'autofl'): continue
    for bb in (0, 1):
        sel = Mk[:, j] & (b == bb)
        slides = np.unique(sl[sel])
        if len(slides) < 2: continue
        for s in slides:
            r = sel & (sl == s); q1, med, q3 = np.percentile(Y[r & (sp == 0) & OKC, j], [25, 50, 75])   # stats on split-0 rows (v3 leak rule)
            Y[r, j] = (Y[r, j] - med) / max((q3 - q1) / 1.349, 1e-6)
        ref = sel & (sp == 0) & OKC; Y[sel, j] = (Y[sel, j] - Y[ref, j].mean()) / Y[ref, j].std()
        print(f'{n:24s} batch {bb}: per-slide robust z over {len(slides)} slides ({sel.sum()} cells)')
Y[~Mk] = 0.0
T['Y_v4'] = T['Y']; T['Y'] = Y.astype(np.float32); T['mask'] = Mk; T['qc_masked'] = qc
np.savez(W + f'/targets_v5{S_}.npz', **T); print(f'wrote targets_v5{S_}.npz')
