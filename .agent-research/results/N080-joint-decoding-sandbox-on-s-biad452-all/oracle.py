import sys
import numpy as np
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/codex')
import train_ifc_sandbox as s
M = s.MARKERS; ix = s.MI
d = np.load('/tmp/n080_rt/fcs_I.npz'); I, S = d['I'], d['S']
c = np.maximum(1.4826*np.median(np.abs(I-np.median(I,0)),0),1.0); Yt = np.arcsinh(I/c)
Yps = np.zeros_like(Yt)
for sv in np.unique(S):
    m = S==sv; Yps[m] = (Yt[m]-Yt[m].mean(0))/(Yt[m].std(0)+1e-6)
donor = s.donor_ids(S)
def fine_types(Y, thr):
    p = Y > thr; P = lambda k: p[:, ix[k]]
    t = np.full(len(Y), 'other', object)
    g = P('CD15'); t[g & P('Siglec8')] = 'eos'; t[g & ~P('Siglec8')] = 'neut'
    ng = ~g; t[ng & P('CD14')] = 'mono'
    l = ng & ~P('CD14')
    t[l & P('CD3') & P('CD4')] = 'CD4T'; t[l & P('CD3') & P('CD8') & ~P('CD4')] = 'CD8T'
    t[l & P('CD3') & ~P('CD4') & ~P('CD8')] = 'T_other'
    t[l & ~P('CD3') & P('CD19')] = 'B'; t[l & ~P('CD3') & ~P('CD19') & P('CD56')] = 'NK'
    return t
for test in (0, 1, 2):
    tr, te = donor == (test+2) % 3, donor == test
    for name, Y in (('per-sample z (pack)', Yps), ('train-donor z only', (Yt - Yt[tr].mean(0)) / Yt[tr].std(0))):
        thr = np.quantile(Y[tr], .75, axis=0)  # simple marginal type gate; any reasonable gate works for the oracle
        # bimodal-ish thresholds: midpoint between q05 and q995 is crude; use q75 for CD15/CD3/CD45-like and q90 for rare
        thr = np.array([np.quantile(Y[tr][:, j], q) for j, q in enumerate([.4, .95, .9, .92, .7, .5, .85, .9, .82])])
        ttr, tte = fine_types(Y[tr], thr), fine_types(Y[te], thr)
        means = {k: Y[tr][ttr == k].mean(0) for k in np.unique(ttr)}
        oracle = np.stack([means[k] for k in tte])
        gates = s.fit_gates(Y[tr])
        mt = s.compute_metrics(Y[te], oracle, gates)
        w = mt['within_population']
        print(f'test{test} {name:20s} TYPE-ORACLE (no within-type info): macro_r {mt["macro_r"]:.3f} within-pop macro r',
              {k: None if v['macro_r'] is None else round(v['macro_r'], 3) for k, v in w.items()},
              'lymph r CD3/CD19/CD4/CD8', [round(w['lymphocytes']['per_marker_r'][m], 2) for m in ('CD3', 'CD19', 'CD4', 'CD8')],
              'frob', round(mt['coexpression']['overall']['frobenius'], 2), 'rmse', round(mt['vector_rmse_mean'], 3),
              'gateF T/B/NK/CD4T', [round(mt['gating'][g]['f1'] or 0, 2) for g in ('T_cells', 'B_cells', 'NK_cells', 'CD4_T')])
