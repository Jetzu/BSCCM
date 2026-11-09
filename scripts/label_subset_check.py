"""N023: do single-marker labels detect the expected positive subset? LF-morphology classes (fluorescence-independent); fraction of the
expected-positive class above the 95th percentile of a negative class (null 0.05). Uses targets_v5."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5.npz',allow_pickle=True); Y,M,sl,sn,b,c3,sp=T['Y'],T['mask'],T['slide'],T['slide_names'].astype(str),T['batch'],T['cls3'],T['split']
names=[str(n) for n in T['names']]
rng=np.random.default_rng(0)
rows=np.sort(np.concatenate([rng.permutation(np.where(sl==s)[0])[:12000] for s in np.unique(sl)]))
F=np.load(W+'/runs/n022_base_fov/probe/lf_feats_all.npy',mmap_mode='r'); X=np.concatenate([np.asarray(F[rows[i:i+20000]],dtype=np.float32) for i in range(0,len(rows),20000)])
Z=PCA(128,random_state=0).fit(X[::4]).transform(X)
tr=(b[rows]==0)&(c3[rows]>=0)&(sp[rows]==0); clf=LogisticRegression(max_iter=3000).fit(Z[tr],c3[rows][tr])
P=clf.predict_proba(Z); cls=np.where(P.max(1)>0.8,P.argmax(1),-1)
# expected positive fractions in PB lymphocytes: CD3 ~65-75%, CD19 ~8-15%, CD56 (NK) ~8-15%; CD14 in monocytes ~90%; HLA-DR in monocytes ~90%
for m,pos,neg in [('CD3',0,1),('CD19',0,1),('CD56',0,1),('CD14',2,0),('HLA-DR',2,1),('CD123',2,1),('CD16',1,0),('CD45',0,1)]:
    j=names.index('single:'+m)
    for bb in (0,1):
        s=[k for k in np.unique(sl) if sn[k]==f'antibodies_{m}/batch_{bb}/slide_replicate_0'][0]
        sel=(sl[rows]==s); y=Y[rows,j]
        thr=np.percentile(y[sel&(cls==neg)],95)
        fp=np.mean(y[sel&(cls==pos)]>thr)
        print(f'{m:7s} b{bb}: frac of {["lym","gran","mono"][pos]} above 95th pct of {["lym","gran","mono"][neg]} = {fp:.3f}  (null 0.05)  n={int((sel&(cls==pos)).sum())}')
