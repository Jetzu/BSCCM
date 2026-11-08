"""N003: Giemsa<->label-free match audit + modality look. Per histology cell: Giemsa cell area (um^2, Otsu on darkness, central component)
vs label-free area from DPC phase image and brightfield; nucleus fraction (dark-purple pixels). Also montage PNGs."""
import numpy as np, pandas as pd, json, sys, os
from multiprocessing import Pool
from skimage.filters import threshold_otsu
from skimage.measure import label
W='/hnvme/workspace/v181bb10-bsccm'; P=W+'/pack'; OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
hc=np.load(P+'/hist_cells.npy')
def central(mask):
    L=label(mask); c=L[L.shape[0]//2-5:L.shape[0]//2+5, L.shape[1]//2-5:L.shape[1]//2+5]
    v=np.bincount(c.ravel()); v[0]=0
    if v.max()==0: return 0
    return int((L==v.argmax()).sum())
def one(r):
    hi=np.load(P+'/histology.npy',mmap_mode='r'); dp=np.load(P+'/dpc.npy',mmap_mode='r'); led=np.load(P+'/led_array.npy',mmap_mode='r')
    g=int(hc[r]); h=hi[r].astype(np.float32); bg=np.percentile(h.reshape(-1,3),90,axis=0); od=-np.log(np.clip(h/(bg+1e-6),1e-3,1.5))
    dark=od.sum(2); a_g=central(dark>max(threshold_otsu(dark),0.15))*0.0534**2
    nuc=(od[...,0]>np.percentile(od[...,0],97)*0.6)  # red-channel absorbance = hematoxylin-like purple
    d=np.abs(dp[g]); a_d=central(d>threshold_otsu(d))*0.166**2
    bf=led[g,0].astype(np.float32); bfd=np.abs(bf-np.median(bf)); a_b=central(bfd>threshold_otsu(bfd))*0.166**2
    df=led[g,9].astype(np.float32)  # DF_70
    return dict(global_index=g,hrow=r,area_giemsa=a_g,area_dpc=a_d,area_bf=a_b,od_mean=float(dark.mean()),od_red=float(od[...,0].mean()),od_blue=float(od[...,2].mean()),df70_mean=float(df.mean()),dpc_absmean=float(d.mean()))
if __name__=='__main__':
    with Pool(int(os.environ.get('NP','48'))) as p: rows=p.map(one,range(len(hc)),chunksize=64)
    df=pd.DataFrame(rows); df.to_csv(OUT+'/giemsa_lf_match.csv',index=False)
    from scipy.stats import spearmanr
    res={k:float(spearmanr(df.area_giemsa,df[k]).correlation) for k in ['area_dpc','area_bf','df70_mean','dpc_absmean']}
    # shuffled control: pair Giemsa with a random other cell's LF
    rng=np.random.default_rng(0); perm=rng.permutation(len(df))
    res.update({k+'_shuffled':float(spearmanr(df.area_giemsa,df[k].values[perm]).correlation) for k in ['area_dpc','area_bf']})
    print(json.dumps(res,indent=1)); json.dump(res,open(OUT+'/giemsa_lf_match_summary.json','w'),indent=1)
    # montage
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    hi=np.load(P+'/histology.npy',mmap_mode='r'); dp=np.load(P+'/dpc.npy',mmap_mode='r'); led=np.load(P+'/led_array.npy',mmap_mode='r'); fl=np.load(P+'/fluor.npy',mmap_mode='r')
    sel=rng.choice(len(hc),8,replace=False); cols=['Giemsa','Brightfield','DPC_Left','DF_70','DPC phase (dpc)','LED119','Fluor 426-446 (autofl.)']
    fig,ax=plt.subplots(8,len(cols),figsize=(2*len(cols),16))
    for i,r in enumerate(sel):
        g=int(hc[r]); h=hi[r].astype(np.float32); h=np.clip(h/np.percentile(h,99.5),0,1)
        ims=[h,led[g,0],led[g,19],led[g,9],dp[g],led[g,22],fl[g,5]]
        for j,im in enumerate(ims):
            ax[i,j].imshow(im if j==0 else im,cmap=None if j==0 else 'gray'); ax[i,j].axis('off')
            if i==0: ax[i,j].set_title(cols[j],fontsize=8)
        ax[i,0].text(5,30,f'g={g}',color='k',fontsize=7)
    plt.tight_layout(); plt.savefig(OUT+'/montage_giemsa_lf.png',dpi=80)
    # LF montage for stained cells per gate class
    T=np.load(W+'/targets.npz'); cls=T['cls']; gates=T['gates']
    fig,ax=plt.subplots(len(gates),6,figsize=(12,2*len(gates)))
    for k in range(len(gates)):
        rr=rng.choice(np.where(cls==k)[0],3,replace=False)
        for i,g in enumerate(rr):
            ax[k,2*i].imshow(led[g,0],cmap='gray'); ax[k,2*i+1].imshow(led[g,9],cmap='gray'); ax[k,2*i].axis('off'); ax[k,2*i+1].axis('off')
        ax[k,0].set_title(str(gates[k]),fontsize=7,loc='left')
    plt.tight_layout(); plt.savefig(OUT+'/montage_gates_bf_df70.png',dpi=70)
    dps=dp[:3000]; print('dpc stats: min %.3f max %.3f mean %.4f' % (dps.min(),dps.max(),dps.mean()))
