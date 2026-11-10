"""Two single-core HGB pairs, no repo mutations or model exports."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
W=Path('/hnvme/workspace/v181bb10-bsccm');T=dict(np.load(W/'targets_v4.npz',allow_pickle=True))
idx=pd.read_parquet(W/'bsccm_index.parquet')
P=idx[['fov_center_x','fov_center_y','fov_center_z','position_in_fov_x_pix','position_in_fov_y_pix']].values.astype(float)
assert np.array_equal(idx.data_path.values,T['slide_names'][T['slide']])
Y,M,sp,h=T['Y_v3'][:,12],T['mask'][:,12],T['split'],T['hist_row']
for s in np.unique(T['slide'][h>=0]):
 m=np.flatnonzero((T['slide']==s)&M);tr=m[sp[m]==0];ng=tr[h[tr]<0]
 def trend(rows):
  model=HGB(max_iter=200,learning_rate=.05,max_leaf_nodes=31,min_samples_leaf=200,random_state=0).fit(P[rows],Y[rows])
  z=model.predict(P[m]);z-=model.predict(P[rows]).mean();return z
 full=trend(tr);clean=trend(ng)
 old=(Y[m]-full).astype(np.float32);new=(Y[m]-clean).astype(np.float32)
 print('slide',int(s),'train all/nonGS/GS',len(tr),len(ng),len(tr)-len(ng))
 print('saved v4 max reproduction error',float(np.max(np.abs(old-T['Y'][m,12]))))
 d=old-new;mm=(h[m]<0)&(sp[m]==0)
 print('nonGS training label changes',int(mm.sum()),'RMSE',float(np.sqrt(np.mean(d[mm]**2))),'old sd',float(old[mm].std()),'corr',float(np.corrcoef(old[mm],new[mm])[0,1]))
 mm=(h[m]>=0)&(sp[m]==2)
 print('GS split2 old-vs-clean truth correlation',float(np.corrcoef(old[mm],new[mm])[0,1]))
 for run in ('n021_gs_anc','n025_tt6_s0'):
  E=np.load(W/'runs'/run/'test_preds.npz');g=E['gidx'];q=T['slide'][g]==s
  loc=np.searchsorted(m,g[q])
  print('student same predictions rescored',run,'v4',float(np.corrcoef(E['student'][q,12],old[loc])[0,1]),'GS-excluded detrend',float(np.corrcoef(E['student'][q,12],new[loc])[0,1]))
print('done')
