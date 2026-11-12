"""N038 assumption breaker C: what does the GOAL metric (pooled Pearson per target x test draw, mean of 11 x 2) reward?
Uses the N031 inductive ensemble (base+zoom2) predictions and targets_v5. Per target x draw: pooled r (reproduce GOAL), Spearman,
winsorised r (labels+preds clipped at 1/99 pct), r without the top-1% cells by |z_p*z_y| contribution, censored share + r w/o censored,
FOV decomposition (within-FOV r on FOV-demeaned P,Y; between-FOV r of FOV means; label FOV-share vs permutation null), FOV-cluster
bootstrap CI of the GOAL mean, composition: r within predicted cls3, and per-slide r for multi-slide grp targets."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True)
Y, M, bt, fov, sl, cz = T['Y'], T['mask'], T['batch'], T['fov'], T['slide'], T['censored']
names = [str(n) for n in T['names']]; C11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G'], E['C3']
rng = np.random.default_rng(0)
def fovdecomp(p, y, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 3
    p, y, inv = p[ok], y[ok], np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    mp, my = np.bincount(inv, p) / c, np.bincount(inv, y) / c
    rw = np.corrcoef(p - mp[inv], y - my[inv])[0, 1]; rb = np.corrcoef(np.repeat(mp, c), np.repeat(my, c))[0, 1]  # cell-weighted
    share = np.var(my[inv]) / np.var(y); share_p = np.var(mp[inv]) / np.var(p)
    nulls = []
    for _ in range(50):
        yy = rng.permutation(y); m_ = np.bincount(inv, yy) / c; nulls.append(np.var(m_[inv]) / np.var(yy))
    return rw, rb, share, float(np.mean(nulls)), share_p, float(np.mean(c))
rows = []; boot_store = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [load(f'n031_{a}_{d}') for a in ('base', 'zoom2')]; G = Ps[0][1]; assert np.array_equal(G, Ps[1][1]); P = np.mean([p[0] for p in Ps], 0); C3 = Ps[0][2]
    for j in C11:
        v = M[G, j]; p, y, g = P[v, j], Y[G[v], j], G[v]
        r = np.corrcoef(p, y)[0, 1]; rs = spearmanr(p, y)[0]
        lo, hi = np.percentile(y, [1, 99]); plo, phi = np.percentile(p, [1, 99]); rwz = np.corrcoef(np.clip(p, plo, phi), np.clip(y, lo, hi))[0, 1]
        zc = ((p - p.mean()) / p.std()) * ((y - y.mean()) / y.std()); keep = np.abs(zc) < np.percentile(np.abs(zc), 99); r_top = np.corrcoef(p[keep], y[keep])[0, 1]
        share_top1 = zc[~keep].sum() / zc.sum()
        c = cz[g, j]; r_nc = np.corrcoef(p[~c], y[~c])[0, 1] if (~c).sum() > 50 else np.nan
        rw, rb, sh, shn, shp, cpf = fovdecomp(p, y, fov[g])
        k3 = C3[v]; within = {kn: (np.corrcoef(p[k3 == k], y[k3 == k])[0, 1] if (k3 == k).sum() > 50 else np.nan) for k, kn in enumerate(['lym', 'gran', 'mono'])}
        frac = {f'frac_{kn}': float((k3 == k).mean()) for k, kn in enumerate(['lym', 'gran', 'mono'])}
        # between-class part: r of predicted-class mean label with label
        mb = np.array([y[k3 == k].mean() if (k3 == k).any() else 0 for k in range(3)])[k3]; r_cls = np.corrcoef(mb, y)[0, 1]
        rows.append(dict(test_draw=b, target=names[j], n=int(v.sum()), r=r, spearman=rs, r_winsor=rwz, r_drop_top1pct=r_top, top1pct_share_of_cov=share_top1,
                         censored=float(c.mean()), r_noncensored=r_nc, r_withinFOV=rw, r_betweenFOV=rb, label_fov_share=sh, label_fov_share_null=shn,
                         pred_fov_share=shp, cells_per_fov=cpf, r_predclass_mean_oracle=r_cls, **{f'r_in_{k}': w for k, w in within.items()}, **frac,
                         skew_y=float(((y - y.mean()) ** 3).mean() / y.std() ** 3)))
        boot_store[(b, j)] = (p, y, fov[g])
        # per-slide r for multi-slide targets
        sls = np.unique(sl[g])
        if len(sls) > 1: rows[-1]['per_slide_r'] = {str(s): float(np.corrcoef(p[sl[g] == s], y[sl[g] == s])[0, 1]) for s in sls}; rows[-1]['slide_mean_pred_diff'] = float(np.diff([p[sl[g] == s].mean() for s in sls])[0])
df = pd.DataFrame(rows)
print('GOAL reproduce', df.groupby('test_draw').r.mean().round(4).to_dict(), 'mean', round(df.r.mean(), 4))
for col in ['spearman', 'r_winsor', 'r_drop_top1pct', 'r_noncensored', 'r_withinFOV']: print(col, round(df[col].mean(), 4))
# FOV-cluster bootstrap of the GOAL mean
B = 1000; bs = np.zeros(B)
for i in range(B):
    rr = []
    for (b, j), (p, y, f) in boot_store.items():
        u = np.unique(f); pick = rng.choice(u, len(u)); idx = np.concatenate([np.flatnonzero(f == q) for q in pick]) if False else None
        # fast: weights = multiplicity of each FOV
        cnt = dict(zip(*np.unique(pick, return_counts=True))); w = np.array([cnt.get(q, 0) for q in f], float)
        pm, ym = np.average(p, weights=w), np.average(y, weights=w); rr.append(np.sum(w * (p - pm) * (y - ym)) / np.sqrt(np.sum(w * (p - pm) ** 2) * np.sum(w * (y - ym) ** 2)))
    bs[i] = np.mean(rr)
print('FOV-bootstrap GOAL mean: sd %.4f, 95%% CI [%.4f, %.4f]' % (bs.std(), *np.percentile(bs, [2.5, 97.5])))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(df[['test_draw', 'target', 'n', 'r', 'spearman', 'r_winsor', 'r_drop_top1pct', 'top1pct_share_of_cov', 'censored', 'r_withinFOV', 'r_betweenFOV', 'label_fov_share', 'label_fov_share_null', 'pred_fov_share', 'cells_per_fov']].round(3).to_string(index=False))
print(df[['test_draw', 'target', 'r', 'r_predclass_mean_oracle', 'r_in_lym', 'r_in_gran', 'r_in_mono', 'frac_lym', 'frac_gran', 'frac_mono', 'skew_y']].round(3).to_string(index=False))
print(df[df.per_slide_r.notna()][['test_draw', 'target', 'r', 'per_slide_r', 'slide_mean_pred_diff']].to_string(index=False) if 'per_slide_r' in df else '')
df.to_csv(sys.argv[1] + '/c_metric.csv', index=False); json.dump(dict(boot_sd=float(bs.std()), ci=list(np.percentile(bs, [2.5, 97.5]))), open(sys.argv[1] + '/c_metric_boot.json', 'w'))
