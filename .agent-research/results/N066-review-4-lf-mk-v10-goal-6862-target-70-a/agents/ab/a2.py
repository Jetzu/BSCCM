import numpy as np, pandas as pd, pyarrow.parquet as pq
W='/hnvme/workspace/v181bb10-bsccm'
cols=pq.read_schema(W+'/bsccm_sm.parquet').names
print([c for c in cols if 'unmixed' in c or 'gated' in c][:60]); print(len(cols))
idx=pd.read_parquet(W+'/bsccm_index.parquet',columns=['antibodies','batch'])
sm=pd.read_parquet(W+'/bsccm_sm.parquet',columns=[c for c in cols if c.endswith('full_model_unmixed') or c.endswith('single_antibody_model_unmixed')])
ab=idx.antibodies.values;b=idx.batch.values
for c in sm.columns:
    x=sm[c].values
    print(c, {f'{a}|{bb}':(round(float(np.isfinite(x[(ab==a)&(b==bb)]).mean()),2)) for a in ['all','CD45','CD16','CD3','unstained'] for bb in (0,1)})
