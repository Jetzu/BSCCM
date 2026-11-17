"""N045 v2 summary: test-split metrics per arm x protocol for PBC original vs re-centred (4,500 steps, seed 0). Writes v2_summary.csv."""
import sys, glob, pandas as pd
rows = []
for f in sorted(glob.glob('/home/hpc/v181bb/v181bb10/bsccm_runs/n045_v2/*/summary.csv')):
    v, p = f.split('/')[-2].split('_', 1); rows.append(pd.read_csv(f).assign(variant=v, run_protocol=p))
d = pd.concat(rows); d.to_csv(sys.argv[1] + '/v2_summary.csv', index=False); pd.set_option('display.width', 250)
r = d[(d.family == 'retrieval') & (d.metric == 'R10') & (d.split == 'test')]
print('R@10 test'); print(r.pivot_table(index='arm', columns=['variant', 'protocol'], values='mean').round(3).to_string())
a = d[(d.family == 'autofl_truth') & (d.metric == 'GS_head_transfer_pearson_r') & (d.split == 'test')]
print('autofl transfer r'); print(a.pivot_table(index='arm', columns=['variant', 'protocol'], values='mean').round(3).to_string())
c = d[(d.family == 'PBC_marker_contrast') & d.protocol.str.startswith('cross')]
print('PBC contrasts mean AUROC'); print(c.pivot_table(index='arm', columns=['variant', 'protocol'], values='mean', aggfunc='mean').round(3).to_string())
k = d[(d.family == 'PBC_sanity') & (d.metric == 'knn_accuracy') & d.protocol.str.startswith('cross')]
print('PBC kNN'); print(k.pivot_table(index='arm', columns=['variant', 'protocol'], values='mean').round(3).to_string())
