"""Aggregate clinical-transfer contrasts (lineage_consistency.json from apply_gs_student.py) over arms x seeds -> long csv + mean/sd table."""
import argparse, json, os, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--arms', nargs='+', required=True, help='arm=dir1,dir2,...')
ap.add_argument('--out', required=True); a = ap.parse_args(); rows = []
for spec in a.arms:
    arm, dirs = spec.split('='); 
    for s, d in enumerate(dirs.split(',')):
        j = json.load(open(os.path.join(a.root, d, 'lineage_consistency.json')))
        for k, v in j['mll23_auroc'].items(): rows.append(dict(arm=arm, seed=s, dir=d, set='MLL23', contrast=k, auroc=v))
        for k, v in j['pbc_lineage_auroc'].items(): rows.append(dict(arm=arm, seed=s, dir=d, set='PBC-B', contrast=k, auroc=v))
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
t = df.groupby(['set', 'contrast', 'arm']).auroc.agg(['mean', 'std', 'count']).round(3).unstack('arm')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100); print(t.to_string())
