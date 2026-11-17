"""N047 AB (b3): (i) which heads does the test-draw readout remap use (full-fit ridge on z-scored heads, HO10)? (ii) does a remap fitted
in the OTHER direction (b10 predictions on draw-0 labels -> applied to b01 on draw 1, and vice versa) transfer? [mild leak: the other
direction's model was trained on this draw's labels; diagnostic only] (iii) in-distribution stacking: remap fitted on IN128 predictions of
the TRAINING draw's split-2 cells (labels of the training draw only) applied to HO10 on the test draw."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt = T['Y'], T['mask'], T['batch']; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return pd.DataFrame(E['P'], index=E['G'])
HO = {d: sum(load(f'{m}_{d}') for m in fam) / len(fam) for d in ('b10', 'b01')}; IN = load('n039_ceil_fov_v5_mt128000')
INd = {b: IN.loc[IN.index.values[bt[IN.index.values] == b]] for b in (0, 1)}
short = [n.replace('single:', '').replace('grp:', 'g:') for n in names]
def fit(P, j):
    m = M[P.index.values, j]; X = P.values[m]; mu, sd = X.mean(0), X.std(0) + 1e-9; y = Y[P.index.values[m], j]
    return Ridge(alpha=max(1.0, .05 * m.sum())).fit((X - mu) / sd, y), mu, sd
def score(P, j, mdl=None, mu=None, sd=None, use_own_norm=True):
    m = M[P.index.values, j]; X = P.values[m]; y = Y[P.index.values[m], j]
    if mdl is None: return np.corrcoef(X[:, j], y)[0, 1]
    if use_own_norm: mu, sd = X.mean(0), X.std(0) + 1e-9   # test-time label-free standardisation of the heads
    return np.corrcoef(mdl.predict((X - mu) / sd), y)[0, 1]
rows = []
for d, b, od in (('b10', 0, 'b01'), ('b01', 1, 'b10')):
    for j in c11:
        mdl, mu, sd = fit(HO[d], j); w = mdl.coef_ / np.abs(mdl.coef_).max()
        top = ' '.join(f'{short[i]}:{w[i]:+.2f}' for i in np.argsort(-np.abs(w))[:4])
        mo, _, _ = fit(HO[od], j); mi, _, _ = fit(INd[1 - b], j)
        rows.append(dict(test_draw=b, target=names[j], raw=score(HO[d], j), full_fit_oracle=score(HO[d], j, mdl), xdir_transfer=score(HO[d], j, mo),
                         indist_stack_from_train_draw=score(HO[d], j, mi), top_weights=top))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/b3_remap_weights.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
print(df.round(3).to_string(index=False)); print(df.groupby('test_draw')[['raw', 'full_fit_oracle', 'xdir_transfer', 'indist_stack_from_train_draw']].mean().round(4).to_string())
print('GOAL-equiv', df[['raw', 'full_fit_oracle', 'xdir_transfer', 'indist_stack_from_train_draw']].mean().round(4).to_dict())
