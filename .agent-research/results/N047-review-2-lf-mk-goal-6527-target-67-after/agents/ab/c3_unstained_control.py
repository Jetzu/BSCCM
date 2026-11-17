"""N047 AB (c3): can the draw-specific autofluorescence coefficient of the single-antibody labels be read from the NEW isolation's
UNSTAINED negative-control slide (no marker label of the new isolation used)? Per marker x draw: recover the authors' unmixing as an
affine map of XS=(total-bg_lowess)/shading on the stained slide (fidelity printed), apply it to the same draw's unstained cells
(true marker = 0) -> pseudo-label; slope of asinh(pseudo/50) on asinh(AF/50) = artefact coefficient on unstained cells. Compare with the
within-LF-class slope on stained cells (authors' unmixed vs AF). Then an inductive-with-negative-control correction:
p' = p + (c_unst(test) - c_unst(train)) * k_train * AFhat, k_train = v5-label units per asinh unit (training draw), AFhat = autofl head
mapped to asinh(AF/50) units on the training draw."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt, hr = T['Y'], T['mask'], T['batch'], T['hist_row']; jaf = names.index('autofl')
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ab = pd.read_parquet(W + '/bsccm_index.parquet', columns=['antibodies']).antibodies.values.astype(str)
CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
TOT = np.stack([sm[f'Fluor_{c}_total_raw'].values for c in CH], 1); BGL = np.stack([sm[f'Fluor_{c}_background_lowess'].values for c in CH], 1); SH = np.stack([sm[f'Fluor_{c}_shading'].values for c in CH], 1)
XS = np.c_[(TOT - BGL) / SH, np.ones(len(TOT))]; AFr = sm['autofluor_single_antibody_model_unmixed'].values
_E = np.load(W + '/runs/n039_ceil_fov_v5_mt128000/eval_preds.npz'); CL = np.full(len(Y), -1); CL[_E['G']] = _E['C3']
def wslope(x, y, c):
    xs = x.copy(); ys = y.copy()
    for k in np.unique(c): s = c == k; xs[s] -= xs[s].mean(); ys[s] -= ys[s].mean()
    return float((xs * ys).sum() / (xs * xs).sum())
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
ENS = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in fam]; ENS[b] = (np.mean([p['P'] for p in Ps], 0), Ps[0]['G'])
info = {}; rows = []
for mk in ['HLA-DR', 'CD19', 'CD14', 'CD45', 'CD123', 'CD56', 'CD3', 'CD16']:
    j = names.index(f'single:{mk}'); un = sm[f'{mk}_single_antibody_model_unmixed'].values
    for b in (0, 1):
        s = np.flatnonzero((ab == mk) & (bt == b) & (hr < 0) & np.isfinite(un) & np.isfinite(AFr))
        beta = np.linalg.lstsq(XS[s], un[s], rcond=None)[0]; fid = np.corrcoef(XS[s] @ beta, un[s])[0, 1]
        ba = np.linalg.lstsq(XS[s], AFr[s], rcond=None)[0]; fida = np.corrcoef(XS[s] @ ba, AFr[s])[0, 1]
        u = np.flatnonzero((ab == 'unstained') & (bt == b) & (hr < 0))
        afu = np.where(np.isfinite(AFr[u]), AFr[u], XS[u] @ ba)
        c_unst = np.polyfit(np.arcsinh(afu / 50), np.arcsinh((XS[u] @ beta) / 50), 1)[0]
        s2 = s[CL[s] >= 0]; c_st = wslope(np.arcsinh(AFr[s2] / 50), np.arcsinh(un[s2] / 50), CL[s2])
        sm_ = s[M[s, j]]; k = np.polyfit(np.arcsinh(un[sm_] / 50), Y[sm_, j], 1)[0]   # v5 units per asinh unit
        ma = sm_[M[sm_, jaf]]; bb, aa = np.polyfit(Y[ma, jaf], np.arcsinh(AFr[ma] / 50), 1)
        info[(mk, b)] = dict(c_unst=c_unst, c_st=c_st, k=k, a=aa, b=bb)
        rows.append(dict(target=mk, draw=b, unmix_fidelity=fid, autofl_fidelity=fida, n_unstained=len(u), c_unstained=c_unst, c_stained_within_class=c_st, k_v5_per_asinh=k))
df = pd.DataFrame(rows); pd.set_option('display.width', 250); print(df.round(3).to_string(index=False))
out = []
for d, te in (('b10', 0), ('b01', 1)):
    tr = 1 - te; P, G = ENS[te]
    for mk in ['HLA-DR', 'CD19', 'CD14', 'CD45', 'CD123', 'CD56', 'CD3', 'CD16']:
        j = names.index(f'single:{mk}'); I_te, I_tr = info[(mk, te)], info[(mk, tr)]
        m = M[G, j] & (ab[G] == mk); y = Y[G[m], j]; p = P[m, j]; afh = I_tr['a'] + I_tr['b'] * P[m, jaf]; afh -= afh.mean()
        dc = (I_te['c_unst'] - I_tr['c_unst']) * I_tr['k']; dc_or = (I_te['c_st'] - I_tr['c_st']) * I_tr['k']
        out.append(dict(dir=d, target=mk, delta_c_unstained=dc, delta_c_stained_ORACLE=dc_or, r_raw=np.corrcoef(p, y)[0, 1],
                        r_negctrl=np.corrcoef(p + dc * afh, y)[0, 1], r_oracle_stained_slope=np.corrcoef(p + dc_or * afh, y)[0, 1]))
o = pd.DataFrame(out); o.to_csv(sys.argv[1] + '/c3_unstained_control.csv', index=False); df.to_csv(sys.argv[1] + '/c3_unstained_slopes.csv', index=False)
print(o.round(3).to_string(index=False))
for c in ('r_negctrl', 'r_oracle_stained_slope'): print(c, 'GOAL delta (/22):', round((o[c] - o.r_raw).sum() / 22, 4), o.assign(dd=o[c] - o.r_raw).groupby('dir').dd.sum().round(4).to_dict())
