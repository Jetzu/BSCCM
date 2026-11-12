import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5.npz',allow_pickle=True); print(T.files)
print([str(n) for n in T['names']]); print('gates', T['gates'])
print('slide_names', list(enumerate(T['slide_names'])))
sm=pd.read_parquet(W+'/bsccm_sm.parquet'); print(sm.shape); print(list(sm.columns))
idx=pd.read_parquet(W+'/bsccm_index.parquet'); print(list(idx.columns)); print(idx.antibodies.value_counts())
F=np.load(W+'/fm_prep.npz',allow_pickle=True); print(F.files, F['stain_names'], F['a_c'])
E=np.load(W+'/runs/n031_base_b10/eval_preds.npz'); print(E.files, E['P'].shape)
