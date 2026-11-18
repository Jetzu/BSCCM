"""N057: TT6c anchor-strength sweep (anchor CE .5 = N056 T arm, 2.0, 8.0), raw (unaligned) students only.
Per unit: macro AUROC summaries (primary16 = pre-registered non-anchor contrasts, MLL23_15, PBC10, all25) + BSCCM same-cell autofl.
Paired by teacher vs anchor .5; noise floor = max within-teacher seed range of either arm (N056 rule).
writes n057_units.csv, n057_summary.csv, n057_paired.csv, plot_n057_primary16.csv, plot_n057_mll23.csv"""
import glob, numpy as np, pandas as pd
R = '/hnvme/workspace/v181bb10-bsccm/runs'
ROOTS = {0.5: f'{R}/n056v2_main', 2.0: f'{R}/n057_a2', 8.0: f'{R}/n057_a8'}
rows = []
for anc, root in ROOTS.items():
    for f in sorted(glob.glob(f'{root}/t*/T_s*/contrasts.csv')):
        c = pd.read_csv(f); c = c[c.arm == 'T']
        s = pd.read_csv(f.replace('contrasts.csv', 'sanity.csv')); s = s[s.arm == 'T']
        t, sd = str(c.teacher.iloc[0]), int(c.seed.iloc[0])
        rows.append(dict(anchor=anc, teacher=t, seed=sd,
                         primary16=c[c.primary == True].auroc.mean(), MLL23_15=c[c.domain == 'MLL23'].auroc.mean(),
                         PBC10=c[c.domain == 'PBC'].auroc.mean(), all25=c.auroc.mean(),
                         n_primary=int((c.primary == True).sum()), n_all=len(c),
                         autofl_slide2=s[s.slide == 2].autofl_r.mean(), autofl_slide19=s[s.slide == 19].autofl_r.mean()))
u = pd.DataFrame(rows); u.to_csv('n057_units.csv', index=False)
M = ['primary16', 'MLL23_15', 'PBC10', 'all25', 'autofl_slide2', 'autofl_slide19']
tm = u.groupby(['anchor', 'teacher'])[M].mean(); rng = u.groupby(['anchor', 'teacher'])[M].agg(lambda x: x.max() - x.min())
summ = tm.groupby('anchor').mean().round(4); summ['n_units'] = u.groupby('anchor').size(); print(summ.to_string()); summ.to_csv('n057_summary.csv')
out = []
for anc in (2.0, 8.0):
    for m in M:
        d = (tm.loc[anc][m] - tm.loc[0.5][m]); floor = max(rng.loc[anc][m].max(), rng.loc[0.5][m].max())
        out.append(dict(comparison=f'anchor {anc} - .5', metric=m, mean_paired=round(float(d.mean()), 4),
                        per_teacher=' '.join(f'{x:+.3f}' for x in d.values), noise_floor=round(float(floor), 4),
                        decision=('better' if d.mean() > floor else 'worse' if d.mean() < -floor else 'no difference')))
p = pd.DataFrame(out); print(p.to_string(index=False)); p.to_csv('n057_paired.csv', index=False)
lab = {0.5: 'anchor .5 (TT6c)', 2.0: 'anchor 2', 8.0: 'anchor 8'}
for m, fn in (('primary16', 'plot_n057_primary16.csv'), ('MLL23_15', 'plot_n057_mll23.csv')):
    pd.DataFrame(dict(arm=u.anchor.map(lab), value=u[m], seed=u.teacher + '_s' + u.seed.astype(str))).to_csv(fn, index=False)
