"""N058: registered N056 seed-floor rule for H_matched (3 seeds x 3 teachers) vs T (7 seeds), raw (non-aligned) students.
Per run: mean AUROC over contrast sets; per teacher: mean over seeds; diff = mean over teachers of (H_matched - T);
floor = max within-teacher seed range of either arm; claim only if |diff| > floor. writes n058_per_run.csv, n058_decision.csv"""
import glob, os, pandas as pd, numpy as np
R = '/hnvme/workspace/v181bb10-bsccm/runs/n058_main'; rows = []
for f in sorted(glob.glob(f'{R}/t*/*/contrasts.csv')):
    run = os.path.basename(os.path.dirname(f)); t = f.split('/')[-3]; arm, s = run.rsplit('_s', 1)
    c = pd.read_csv(f); c = c[c.arm == arm]          # raw student rows only (aligned variants carry suffixed arm names)
    rows.append(dict(arm=arm, teacher=t, seed=int(s), MLL23_15=c[c.domain == 'MLL23'].auroc.mean(),
                     primary16=c[c.primary.astype(bool)].auroc.mean(), PBC10=c[c.domain == 'PBC'].auroc.mean(), all25=c.auroc.mean(), n=len(c)))
d = pd.DataFrame(rows); d.to_csv('n058_per_run.csv', index=False); print(d.round(3).to_string(index=False))
out = []
for m in ['MLL23_15', 'primary16', 'all25', 'PBC10']:
    tm = d.groupby(['arm', 'teacher'])[m].mean().unstack(0); rng = d.groupby(['arm', 'teacher'])[m].agg(lambda x: x.max() - x.min()).unstack(0)
    diff = float((tm['H_matched'] - tm['T']).mean()); floor = float(rng.max().max())
    out.append(dict(metric=m, T=float(tm['T'].mean()), H_matched=float(tm['H_matched'].mean()), diff=diff, floor=floor,
                    per_teacher=' '.join(f'{x:+.3f}' for x in (tm['H_matched'] - tm['T'])),
                    decision='no difference' if abs(diff) <= floor else ('H_matched higher' if diff > 0 else 'T higher')))
o = pd.DataFrame(out); o.to_csv('n058_decision.csv', index=False); print(o.round(3).to_string(index=False))
