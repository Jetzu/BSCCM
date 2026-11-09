"""N012: spatial-confounding control for within-slide GS/LF -> autofl on the 15,712 Wright cells.
Compares cell-level random KFold vs GroupKFold by FOV vs GroupKFold by slide region (4x4 blocks of FOV centres), for: position-only
features (FOV centre x/y/z + position in FOV), frozen DinoBloom-B GS features, frozen sup-LF features; plus cross-slide numbers."""
import numpy as np, pandas as pd, json, sys
from sklearn.linear_model import RidgeCV; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingRegressor
W='/hnvme/workspace/v181bb10-bsccm'; OUT=sys.argv[1]; LFD=sys.argv[2]
T=np.load(W+'/targets_v3.npz'); idx=pd.read_parquet(W+'/bsccm_index.parquet'); j=list(T['names']).index('autofl')
hrow=T['hist_row']; gidx=np.where(hrow>=0)[0]; gidx=gidx[np.argsort(hrow[gidx])]; gb=T['batch'][gidx]; y=T['Y'][gidx,j]
P=idx.loc[gidx,['fov_center_x','fov_center_y','fov_center_z','position_in_fov_x_pix','position_in_fov_y_pix']].values.astype(float)
fov=T['fov'][gidx]; GS=np.load(W+'/emb/giemsa_dinobloom_B.npy'); LF=np.load(LFD+'/lf_feats_all.npy').astype(np.float32)[gidx]
def region(m, k=4):
    x,yy=P[m,0],P[m,1]; qx=np.searchsorted(np.quantile(x,np.linspace(0,1,k+1)[1:-1]),x); qy=np.searchsorted(np.quantile(yy,np.linspace(0,1,k+1)[1:-1]),yy); return qx*k+qy
pr=lambda p,t: float(np.corrcoef(p,t)[0,1]); res={}
ridge=lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,8)))
for bb in (0,1):
    m=gb==bb; groups={'cell_random':None,'fov_grouped':fov[m],'region_4x4_blocked':region(m)}
    for gname,g in groups.items():
        cv=KFold(5,shuffle=True,random_state=0) if g is None else GroupKFold(5)
        for fname,X,mk in (('position_only_GBM',P[m],lambda: HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05)),('GS_dinobloom_ridge',GS[m],ridge),('LF_sup_ridge',LF[m],ridge),('GS+position_ridge',np.c_[GS[m],P[m]],ridge)):
            pred=cross_val_predict(mk(),X,y[m],cv=cv,groups=g); res[f'b{bb}/{gname}/{fname}']=pr(pred,y[m])
        print(bb,gname,{k.split('/')[-1]:round(v,3) for k,v in res.items() if k.startswith(f'b{bb}/{gname}')},flush=True)
for bb in (0,1):
    m,o=gb==bb,gb!=bb
    for fname,X in (('GS_dinobloom_ridge',GS),('LF_sup_ridge',LF)):
        res[f'cross_slide_train_b{bb}_test_b{1-bb}/{fname}']=pr(ridge().fit(X[m],y[m]).predict(X[o]),y[o])
json.dump(res,open(OUT+'/spatial_control.json','w'),indent=1); print(json.dumps({k:round(v,3) for k,v in res.items() if 'cross' in k},indent=0))
