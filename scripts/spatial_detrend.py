"""N012b: (1) position-only predictability of EVERY target on EVERY slide (marker spatial-artefact audit), region-blocked CV;
(2) spatially detrended autofl on the GS slides (cross-fitted GBM on position, region-blocked) and GS/LF -> residual, within (blocked)
and cross-slide."""
import numpy as np, pandas as pd, json, sys
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import RidgeCV; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
W='/hnvme/workspace/v181bb10-bsccm'; OUT=sys.argv[1]; LFD=sys.argv[2]
T=np.load(W+'/targets_v3.npz'); idx=pd.read_parquet(W+'/bsccm_index.parquet'); names=list(T['names']); Y,Mk=T['Y'],T['mask']
P=idx[['fov_center_x','fov_center_y','fov_center_z','position_in_fov_x_pix','position_in_fov_y_pix']].values.astype(float); slide=idx.data_path.values
def region(Pm,k=4):
    x,y=Pm[:,0],Pm[:,1]; return np.searchsorted(np.quantile(x,np.linspace(0,1,k+1)[1:-1]),x)*k+np.searchsorted(np.quantile(y,np.linspace(0,1,k+1)[1:-1]),y)
pr=lambda p,t: float(np.corrcoef(p,t)[0,1]); res={}; rng=np.random.default_rng(0)
gbm=lambda: HGB(max_iter=150,learning_rate=0.05)
# (1) audit
for s in sorted(set(slide)):
    for j,n in enumerate(names):
        m=np.where((slide==s)&Mk[:,j])[0]
        if len(m)<2000: continue
        m=rng.permutation(m)[:15000]
        p=cross_val_predict(gbm(),P[m],Y[m,j],cv=GroupKFold(5),groups=region(P[m])); res[f'posonly_blocked/{n}/{s}']=pr(p,Y[m,j])
print('position-only (region-blocked) r, max over slides per target:', {n:round(max([v for k,v in res.items() if k.split('/')[1]==n] or [np.nan]),3) for n in names},flush=True)
print('median over slides:', {n:round(float(np.median([v for k,v in res.items() if k.split('/')[1]==n] or [np.nan])),3) for n in names},flush=True)
# (2) detrended autofl on GS cells
j=names.index('autofl'); hrow=T['hist_row']; gidx=np.where(hrow>=0)[0]; gidx=gidx[np.argsort(hrow[gidx])]; gb=T['batch'][gidx]; y=Y[gidx,j]
GS=np.load(W+'/emb/giemsa_dinobloom_B.npy'); LF=np.load(LFD+'/lf_feats_all.npy').astype(np.float32)[gidx]; Pg=P[gidx]
resid=np.zeros_like(y)
for bb in (0,1):
    m=np.where(gb==bb)[0]; resid[m]=y[m]-cross_val_predict(gbm(),Pg[m],y[m],cv=GroupKFold(5),groups=region(Pg[m]))
ridge=lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,8)))
for bb in (0,1):
    m=gb==bb; o=gb!=bb; g=region(Pg[m])
    for fn,X in (('GS',GS),('LF',LF)):
        res[f'detrended_autofl/within_blocked/{fn}/b{bb}']=pr(cross_val_predict(ridge(),X[m],resid[m],cv=GroupKFold(5),groups=g),resid[m])
        res[f'detrended_autofl/cross_train_b{bb}_test_b{1-bb}/{fn}']=pr(ridge().fit(X[m],resid[m]).predict(X[o]),resid[o])
        res[f'detrended_autofl/position_r_of_features/{fn}/b{bb}']=float(np.mean([pr(cross_val_predict(ridge(),X[m],Pg[m,c],cv=GroupKFold(5),groups=g),Pg[m,c]) for c in (0,1)]))
print({k:round(v,3) for k,v in res.items() if k.startswith('detrended')},flush=True)
np.save(W+'/gs_autofl_detrended.npy',resid); json.dump(res,open(OUT+'/spatial_detrend.json','w'),indent=1)
