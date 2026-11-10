"""N033: label file matching the TT6c two-teacher chain: markers from targets_v4c (clean calibration), autofl from targets_v6c
(unstained slides only, clean). Used as measured labels for the virtual-stain data engine. Writes targets_tt6c.npz."""
import numpy as np
W = '/hnvme/workspace/v181bb10-bsccm'
A = dict(np.load(W + '/targets_v4c.npz', allow_pickle=True)); B = np.load(W + '/targets_v6c.npz', allow_pickle=True)
names = [str(n) for n in A['names']]; assert names == [str(n) for n in B['names']]; j = names.index('autofl')
for k in ('hist_row', 'split', 'batch', 'slide'): assert np.array_equal(A[k], B[k])
Y, M = A['Y'].copy(), A['mask'].copy(); Y[:, j], M[:, j] = B['Y'][:, j], B['mask'][:, j]
A['Y'], A['mask'] = Y, M; A['slide_names'] = A['slide_names'].astype(str); np.savez(W + '/targets_tt6c.npz', **A)
print('autofl labelled', int(M[:, j].sum()), '| any label', int(M.any(1).sum()))
