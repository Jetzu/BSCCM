"""N008b: in-domain morphology anchor. k-means (k=12) on frozen DinoBloom-B embeddings of BSCCM Wright cells (PCA-50 whitened);
montage of 10 random members per cluster for visual morphology labelling (non-expert; to be confirmed by hematologists)."""
import numpy as np, json, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
W='/hnvme/workspace/v181bb10-bsccm'; OUT=sys.argv[1]; K=int(sys.argv[2]) if len(sys.argv)>2 else 12
E=np.load(W+'/emb/giemsa_dinobloom_B.npy'); Z=PCA(50,whiten=True,random_state=0).fit_transform(E)
km=KMeans(K,n_init=10,random_state=0).fit(Z); lab=km.labels_; np.save(OUT+f'/gs_kmeans{K}.npy',lab)
h=np.load(W+'/pack/histology.npy',mmap_mode='r'); sc=json.load(open(W+'/emb/scale.json'))['scale_ref']; T=np.load(W+'/targets_v3.npz')
gidx=np.where(T['hist_row']>=0)[0]; gidx=gidx[np.argsort(T['hist_row'][gidx])]; b=T['batch'][gidx]
rng=np.random.default_rng(0); fig,ax=plt.subplots(K,10,figsize=(15,1.6*K))
for k in range(K):
    mem=np.where(lab==k)[0]
    for i,r in enumerate(rng.choice(mem,10,replace=False)):
        ax[k,i].imshow(np.clip(np.asarray(h[r],np.float32)/sc,0,1)); ax[k,i].axis('off')
    ax[k,0].set_title(f'c{k} n={len(mem)} ({len(mem)/len(lab)*100:.1f}%) b0={np.mean(b[mem]==0)*100:.0f}%',fontsize=8,loc='left')
plt.tight_layout(); plt.savefig(OUT+f'/gs_kmeans{K}_montage.png',dpi=55); print({k:int((lab==k).sum()) for k in range(K)})
