"""N012: where is the GS->molecular signal lost across the two stained slides?  All ridge on frozen features (CPU)."""
import numpy as np, json, sys, pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression; from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
W='/hnvme/workspace/v181bb10-bsccm'; OUT=sys.argv[1]; LFD=sys.argv[2]
T=np.load(W+'/targets_v3.npz'); idx=pd.read_parquet(W+'/bsccm_index.parquet'); sm=pd.read_parquet(W+'/bsccm_sm.parquet')
hrow,b,sp,Y,Mk=T['hist_row'],T['batch'],T['split'],T['Y'],T['mask']; j=list(T['names']).index('autofl')
gidx=np.where(hrow>=0)[0]; gidx=gidx[np.argsort(hrow[gidx])]; slide=idx.data_path.values
GS=np.load(W+'/emb/giemsa_dinobloom_B.npy'); LF=np.load(LFD+'/lf_feats_all.npy').astype(np.float32)
y=Y[gidx,j]; gb=b[gidx]; res={}
def pr(p,t): return float(np.corrcoef(p,t)[0,1])
# 1) autofl raw + calibrated per unstained slide
raw=sm['autofluor_single_antibody_model_unmixed'].values
for s in sorted(set(slide[idx.antibodies=='unstained'])):
    m=slide==s; h=m&(hrow>=0); res[f'autofl_raw_median/{s}']=float(np.nanmedian(raw[m])); res[f'autofl_z_mean/{s}']=float(Y[m&Mk[:,j],j].mean()); res[f'n_hist/{s}']=int(h.sum())
# 2) LF teacher (ridge on stained+unstained NON-GS training cells' autofl) evaluated per GS slide
tr=np.where((sp==0)&Mk[:,j]&(hrow<0))[0]; tr=np.random.default_rng(0).permutation(tr)[:80000]
rt=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,8))).fit(LF[tr],Y[tr,j]); tp=rt.predict(LF[gidx])
for bb in (0,1): res[f'teacher_LF_r_on_GS_slide_b{bb}']=pr(tp[gb==bb],y[gb==bb])
res['teacher_LF_r_on_both_GS_slides_pooled']=pr(tp,y)
# 3) direct GS->autofl ridge: within-slide 5-fold CV vs cross-slide
for bb in (0,1):
    m=gb==bb; res[f'direct_GS_withinslide_cv_r_b{bb}']=pr(cross_val_predict(make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,8))),GS[m],y[m],cv=KFold(5,shuffle=True,random_state=0)),y[m])
    o=gb!=bb; mdl=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,8))).fit(GS[m],y[m]); res[f'direct_GS_train_b{bb}_test_b{1-bb}_r']=pr(mdl.predict(GS[o]),y[o])
# same for LF features restricted to GS cells (does LF shift too?)
for bb in (0,1):
    m=gb==bb; o=gb!=bb; mdl=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,8))).fit(LF[gidx][m],y[m]); res[f'direct_LF_onGScells_train_b{bb}_test_b{1-bb}_r']=pr(mdl.predict(LF[gidx][o]),y[o])
    res[f'direct_LF_onGScells_withinslide_cv_r_b{bb}']=pr(cross_val_predict(make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(0,5,8))),LF[gidx][m],y[m],cv=KFold(5,shuffle=True,random_state=0)),y[m])
# 4) domain separability of the two stained slides
for nm,X in (('GS_dinobloom',GS),('LF_sup',LF[gidx])):
    res[f'slide_classifier_acc/{nm}']=float(cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000)),X,gb,cv=5).mean())
json.dump(res,open(OUT+'/diag_slide_shift.json','w'),indent=1); print(json.dumps(res,indent=1))
