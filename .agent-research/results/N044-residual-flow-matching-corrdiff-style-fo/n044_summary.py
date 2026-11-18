"""N044: mean over every (stain, target) entry of each eval set for residual flow (cfg 1, 2) and the plain flow (same harness, cfg 2).
writes n044_summary.csv (long: arm, set, metric, value) + per-entry n044_entries.csv"""
import json, numpy as np, pandas as pd
R = '/hnvme/workspace/v181bb10-bsccm/runs/n044'; arms = {'residual flow g1': 'eval_s48_g1.0', 'residual flow g2': 'eval_s48_g2.0', 'plain flow g2': 'eval_plainflow_s48_g2.0'}
M = ['r_reg', 'r_avgread', 'crps_reg', 'crps_sample', 'frac_outside', 'sd_ratio_sample', 'condW1_sample', 'condW1_reg']; rows = []
for arm, d in arms.items():
    E = json.load(open(f'{R}/{d}/eval.json'))['sets']
    for st, per in E.items():
        for stain, tg in per.items():
            for t, v in tg.items():
                if isinstance(v, dict): rows.append(dict(arm=arm, set=st, stain=stain, target=t, **{m: v.get(m) for m in M}))
df = pd.DataFrame(rows); df.to_csv('n044_entries.csv', index=False)
s = df.groupby(['arm', 'set'])[M].mean().round(4); print(s.to_string()); s.reset_index().to_csv('n044_summary.csv', index=False)
