"""N054: per-arm summary of the frozen primary table (runs/n054v3_tables): M1 autofl truth, M2 r_est (fixed ridge teacher, 12
markers, CIA estimate), M5 LF->MK r of each arm's own LF path (same draw), M6 clinical AUROC means (PBC, MLL23), per holdout + mean.
writes n054_arm_summary.csv"""
import pandas as pd, numpy as np
T = '/hnvme/workspace/v181bb10-bsccm/runs/n054v3_tables'; d = pd.read_csv(f'{T}/primary_metrics.csv')
rows = []
for (h, arm), g in d.groupby(['holdout', 'arm']):
    m2 = g[(g.metric == 'M2') & (g.path == 'fixed_ridge') & (g.target != 'autofl')]
    m2s = g[(g.metric == 'M2') & (g.path == 'fixed_ridge') & (g.target != 'autofl')]
    m5 = g[(g.metric == 'M5_same_draw') & (g.path == 'own_LF') & (g.target != 'autofl')]
    m6 = g[g.metric == 'M6']
    rows.append(dict(holdout=h, arm=arm, M1_autofl=g[g.metric == 'M1'].r.mean(),
                     M2_r_student_teacher=m2.r_student_teacher.mean(), M2_r_est=m2.r_est.mean(), M5_own_LF_r=m5.r.mean(),
                     M6_PBC_auroc=m6[m6.dataset.astype(str).str.upper().str.contains('PBC')].auroc.mean(),
                     M6_MLL23_auroc=m6[m6.dataset.astype(str).str.upper().str.contains('MLL')].auroc.mean(), n_M6=len(m6)))
s = pd.DataFrame(rows); m = s.groupby('arm').mean(numeric_only=True).reset_index().assign(holdout='mean')
out = pd.concat([s, m]).round(3); out.to_csv('n054_arm_summary.csv', index=False); print(out.to_string(index=False))
