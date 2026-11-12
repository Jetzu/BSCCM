"""N035: tabulate the frozen evals (eval_split1 primary, eval_split2 secondary) per label + means over valid labels; apply the
pre-registered decision rule to the PRIMARY cohort (newdraw = draw 1 split 1). Usage: n035_summary.py OUTDIR"""
import json, sys, numpy as np, pandas as pd
D = '/hnvme/workspace/v181bb10-bsccm/runs/n035/'; rows = []
for ev in ('eval_split1', 'eval_split2'):
    E = json.load(open(D + ev + '/eval.json'))
    for cohort, S in E['sets'].items():
        for st, v in S.items():
            for lab, e in v.items():
                if isinstance(e, dict) and 'r_avgread' in e:
                    rows.append(dict(eval=ev, cohort=cohort, stain=st, label=lab, **{k: e.get(k, np.nan) for k in
                        ('n', 'r_realref', 'r_avgread', 'r_reg', 'r_shuffled', 'crps_sample', 'crps_reg', 'condw1_sample', 'condw1_reg',
                         'condw1_shuffled', 'w1_sample', 'w1_reg', 'w1_shuffled', 'frac_outside', 'ideal_outside', 'sd_ratio_sample')}))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n035_per_label.csv', index=False)
M = df.groupby(['eval', 'cohort']).agg(n_labels=('label', 'size'), **{c: (c, 'mean') for c in df.columns[5:] if c != 'n'}).round(3)
M.to_csv(sys.argv[1] + '/n035_means.csv'); pd.set_option('display.width', 250); print(M.T.to_string())
p = M.loc[('eval_split1', 'newdraw')]
v = dict(n_labels=int(p.n_labels), distributionally_better=bool(p.crps_sample < p.crps_reg and p.condw1_sample < p.condw1_reg),
         not_better_per_cell=bool(p.r_avgread <= p.r_reg + .01), calibrated=bool(p.frac_outside <= 2 / 9 + .05),
         crps=(p.crps_sample, p.crps_reg), condw1=(p.condw1_sample, p.condw1_reg), r=(p.r_avgread, p.r_reg), outside=p.frac_outside)
print('PRIMARY VERDICT', v); json.dump(v, open(sys.argv[1] + '/n035_verdict.json', 'w'), indent=1, default=float)
print(df[(df['eval'] == 'eval_split1') & (df.cohort == 'newdraw')][['label', 'r_realref', 'r_avgread', 'r_reg', 'r_shuffled', 'crps_sample', 'crps_reg', 'condw1_sample', 'condw1_reg', 'frac_outside']].round(3).to_string(index=False))
