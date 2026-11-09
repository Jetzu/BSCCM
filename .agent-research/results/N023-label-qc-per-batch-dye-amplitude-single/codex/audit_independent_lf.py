"""Sampled LF-class checks with SSL-only and v4-supervised feature encoders."""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
W=Path('/hnvme/workspace/v181bb10-bsccm'); O=Path(__file__).resolve().parent
T=np.load(W/'targets_v4.npz',allow_pickle=True); U=np.load(W/'targets_v5.npz',allow_pickle=True)
Y,M,sl,sn,b,c3,sp=[T[k] for k in ('Y','mask','slide','slide_names','batch','cls3','split')]; names=list(T['names']); ja=names.index('grp:CD123/HLA-DR/CD14')
rng=np.random.default_rng(23); allslides=[s for s,n in enumerate(sn) if n.startswith('antibodies_all/')]
rows=np.sort(np.concatenate([np.flatnonzero(sl==s) if s==8 else rng.permutation(np.flatnonzero(sl==s))[:(5000 if s in allslides else 1200)] for s in np.unique(sl)]))
tr=(b[rows]==0)&(c3[rows]>=0)&(sp[rows]==0); ev=(b[rows]==0)&(c3[rows]>=0)&(sp[rows]==2)
for route,path in [('SSL_only','runs/n007_ssl_aug/probe_fov/lf_feats_all.npy'),('v4_supervised','runs/n022_base_fov/probe/lf_feats_all.npy')]:
    F=np.load(W/path,mmap_mode='r'); X=np.asarray(F[rows],dtype=np.float32)
    pca=PCA(128,random_state=0).fit(X[tr]); Z=pca.transform(X); del X
    clf=LogisticRegression(max_iter=2500).fit(Z[tr],c3[rows][tr]); p=clf.predict_proba(Z); pred=p.argmax(1); k=np.where(p.max(1)>.8,pred,-1)
    print('ROUTE',route,'feature shape',F.shape,'sample',len(rows),'b0 heldout FOV accuracy',clf.score(Z[ev],c3[rows][ev]),'balanced',balanced_accuracy_score(c3[rows][ev],pred[ev]),'confusion',confusion_matrix(c3[rows][ev],pred[ev]).tolist(),flush=True)
    for s in allslides:
        for split in ['all','test']:
            r=(sl[rows]==s)&((sp[rows]==2) if split=='test' else True); g=rows[r]; kk=k[r]; y1=Y[g[kk==2],ja]; y0=Y[g[kk==0],ja]
            d=(y1.mean()-y0.mean())/np.sqrt((y1.var()+y0.var())/2)
            print('LF CONTRAST',route,sn[s],split,'counts',[int((kk==c).sum()) for c in range(3)],'means',[float(Y[g[kk==c],ja].mean()) for c in range(3)],'d',d,flush=True)
    if route=='SSL_only':
        for s in np.unique(sl):
            r=(sl[rows]==s)&M[rows,-1]; g=rows[r]; kk=k[r]
            if not len(g): continue
            print('AUTOFL LF CLASSES',sn[s],'confident proportions',[float((kk==c).mean()) for c in range(3)],'v4means',[float(Y[g[kk==c],-1].mean()) for c in range(3)],'v5means',[float(U['Y'][g[kk==c],-1].mean()) for c in range(3)],flush=True)
