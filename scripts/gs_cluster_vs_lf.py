"""N008b: crosstab of Wright-morphology clusters (DinoBloom k-means) vs LF-predicted cell class of the SAME cells
(logistic regression on frozen supervised-LF features, trained on gated batch-0 'all' cells of the training split)."""
import numpy as np, sys, json
from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
W='/hnvme/workspace/v181bb10-bsccm'; OUT=sys.argv[1]; F=sys.argv[2]
T=np.load(W+'/targets_v3.npz'); X=np.load(F+'/lf_feats_all.npy').astype(np.float32); c3,sp,hrow=T['cls3'],T['split'],T['hist_row']
tr=np.where((c3>=0)&(sp==0))[0]; te=np.where((c3>=0)&(sp==2))[0]
clf=make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=3000)).fit(X[tr],c3[tr]); acc=float((clf.predict(X[te])==c3[te]).mean())
gidx=np.where(hrow>=0)[0]; gidx=gidx[np.argsort(hrow[gidx])]; P=clf.predict_proba(X[gidx]); pred=P.argmax(1)
lab=np.load(OUT+'/gs_kmeans12.npy'); M=np.zeros((lab.max()+1,3),int)
for k,p in zip(lab,pred): M[k,p]+=1
res={'lf_cls3_test_acc':acc,'gs_cells_pred_composition':dict(zip(['lym','gran','mono'],np.bincount(pred,minlength=3)/len(pred))),
     'cluster_x_lfpred_rows=cluster_cols=[lym,gran,mono]':M.tolist(),'cluster_majority':{int(k):['lym','gran','mono'][int(M[k].argmax())]+f' {M[k].max()/M[k].sum():.2f}' for k in range(len(M))}}
np.savez(OUT+'/gs_lf_cls3_pred.npz',gidx=gidx,pred=pred,P=P); json.dump(res,open(OUT+'/gs_cluster_vs_lf.json','w'),indent=1,default=float); print(json.dumps(res,indent=0,default=float))
