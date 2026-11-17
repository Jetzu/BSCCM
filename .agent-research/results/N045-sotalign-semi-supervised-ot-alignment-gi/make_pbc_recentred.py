"""N045 red-team M3 variant: shift PBC DinoBloom-B features so PBC half A (even rows, unlabelled use only) has the mean of the
TRAINING draw's split-0 BSCCM Giemsa features (the CCA teacher is fitted/centred there). One file per training draw; labels untouched.
Usage: make_pbc_recentred.py OUTDIR"""
import sys, json, numpy as np
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5c.npz', allow_pickle=True)
G = np.load(W + '/emb/giemsa_dinobloom_B.npy').astype(np.float64); P = np.load(W + '/emb/pbc_dinobloom_B.npy').astype(np.float64)
hc = np.load(W + '/pack/hist_cells.npy'); assert np.array_equal(T['hist_row'][hc], np.arange(len(hc))); b, s = T['batch'][hc], T['split'][hc]
muA = P[0::2].mean(0); info = {}
for d in (0, 1):
    muG = G[(b == d) & (s == 0)].mean(0); Q = (P - muA + muG).astype(np.float32)
    np.save(f'{sys.argv[1]}/pbc_dinobloom_B_recentred_b{d}.npy', Q)
    info[f'b{d}'] = dict(n_giemsa_train=int(((b == d) & (s == 0)).sum()), shift_norm=float(np.linalg.norm(muG - muA)), pbc_norm=float(np.linalg.norm(muA)))
json.dump(info, open(f'{sys.argv[1]}/pbc_recentred_info.json', 'w'), indent=1); print(info)
