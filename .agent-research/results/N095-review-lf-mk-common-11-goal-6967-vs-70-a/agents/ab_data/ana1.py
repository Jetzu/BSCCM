import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
Y,M,C=T['Y'],T['mask'],T['censored']; names=[str(n) for n in T['names']]; sn=[str(s) for s in T['slide_names']]
sl,fov=T['slide'],T['fov']
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def r(a,b): return np.corrcoef(a,b)[0,1]
def demean(x,g):
    u,inv=np.unique(g,return_inverse=True); s=np.bincount(inv,x); n=np.bincount(inv); return x-(s/n)[inv], (s/n)[inv]
rows=[]; slrows=[]
for d in ('b10','b01'):
    z=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=z['P'],z['G']
    for j in c11:
        k=M[G,j]; p=P[k,j]; g=G[k]; y=Y[g,j]; c=C[g,j]; s=sl[g]; f=fov[g]
        base=r(p,y)
        # within-slide centering both
        pw,_=demean(p,s); yw,_=demean(y,s); r_ws=r(pw,yw)
        # per-slide z of predictions (training-free, uses test inputs only)
        pz=p.copy()
        for u in np.unique(s):
            m=s==u; pz[m]=(p[m]-p[m].mean())/p[m].std()
        r_pz=r(pz,y)
        # per-slide affine oracle (labels used)
        pa=p.copy()
        for u in np.unique(s):
            m=s==u; A=np.c_[np.ones(m.sum()),p[m]]; pa[m]=A@np.linalg.lstsq(A,y[m],rcond=None)[0]
        r_aff=r(pa,y)
        # within-FOV centering both
        pf,pfm=demean(p,f); yf,yfm=demean(y,f); r_wf=r(pf,yf)
        # FOV-mean oracle: give the model the true FOV means
        r_fovor=r(pf+yfm,y)
        # label FOV-offset removed: y - FOV mean(y) + FOV mean(p)  (i.e. labels' FOV-level part replaced by prediction's)
        r_lab_fovrm=r(p,yf+pfm)
        # censoring
        nc=~c; r_unc=r(p[nc],y[nc]) if nc.sum()>10 else np.nan
        # affine residual share from censored cells
        A=np.c_[np.ones(len(p)),p]; res=y-A@np.linalg.lstsq(A,y,rcond=None)[0]
        cshare=(res[c]**2).sum()/(res**2).sum() if c.any() else 0
        # censoring-consistent oracle: censored cell whose calibrated pred is below its (floor) label counts as exact
        yhat=A@np.linalg.lstsq(A,y,rcond=None)[0]; y2=y.copy(); m=c&(yhat<y); y2[m]=yhat[m]; r_cens=r(p,y2)
        rows.append(dict(dir=d,target=names[j],n=len(p),nslides=len(np.unique(s)),nfov=len(np.unique(f)),r=base,within_slide=r_ws,pred_slide_z=r_pz,slide_affine_oracle=r_aff,
                         within_fov=r_wf,fov_mean_oracle=r_fovor,label_fov_offset_removed=r_lab_fovrm,cens_frac=c.mean(),r_uncensored=r_unc,cens_resid_share=cshare,cens_consistent=r_cens))
        for u in np.unique(s):
            m=s==u
            slrows.append(dict(dir=d,target=names[j],slide=sn[u],n=m.sum(),r=r(p[m],y[m]),y_mean=y[m].mean(),y_sd=y[m].std(),p_mean=p[m].mean(),p_sd=p[m].std(),cens=c[m].mean(),
                               r_unc=r(p[m&nc],y[m&nc]) if (m&nc).sum()>10 else np.nan, within_fov=r(demean(p[m],f[m])[0],demean(y[m],f[m])[0])))
df=pd.DataFrame(rows); sd=pd.DataFrame(slrows)
df.to_csv('/tmp/n095_ab/ab_oracles_per_target.csv',index=False); sd.to_csv('/tmp/n095_ab/ab_per_slide.csv',index=False)
pd.set_option('display.width',250)
print(df.round(3).to_string(index=False))
cols=['r','within_slide','pred_slide_z','slide_affine_oracle','within_fov','fov_mean_oracle','label_fov_offset_removed','r_uncensored','cens_consistent']
g=df.groupby('dir')[cols].mean(); g.loc['GOAL']=g.mean(); print(g.round(4).to_string())
g.to_csv('/tmp/n095_ab/ab_oracles_goal.csv')
print(sd[sd.nslides if False else sd.target.str.startswith('grp')].round(3).to_string(index=False))
