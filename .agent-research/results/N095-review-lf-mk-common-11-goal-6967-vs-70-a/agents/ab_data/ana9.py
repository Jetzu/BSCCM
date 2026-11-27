import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
def r(a,b): return np.corrcoef(a,b)[0,1]
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; fov=T['fov']
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
E={d:np.load(f'/tmp/n095_ab/e8_{d}.npz') for d in ('b10','b01')}
ufull,cfull=np.unique(fov,return_counts=True); full=dict(zip(ufull,cfull))
def loo(p,f):
    u,inv=np.unique(f,return_inverse=True); s=np.bincount(inv,p); n=np.bincount(inv); nn=n[inv]
    m=np.where(nn>1,(s[inv]-p)/np.maximum(nn-1,1),np.nan); return m,nn
res={}
for d in ('b10','b01'):
    P,G=E[d]['P'],E[d]['G']; f=fov[G]; u,c=np.unique(f,return_counts=True)
    print(d,'FOVs complete in test set:',np.mean([full[x]==k for x,k in zip(u,c)]).round(3),'cells/FOV mean',c.mean().round(2))
    for j in c11:
        k=M[G,j]; p=P[k,j]; y=Y[G[k],j]; ff=f[k]
        lm,nn=loo(p,ff); ok=~np.isnan(lm)
        A=np.c_[np.ones(len(p)),p]; e=y-A@np.linalg.lstsq(A,y,rcond=None)[0]
        X=np.c_[np.ones(ok.sum()),p[ok],lm[ok],np.log(nn[ok])]
        w=np.linalg.lstsq(X,y[ok],rcond=None)[0]
        res[(d,names[j])]=dict(r_resid_loo=r(e[ok],lm[ok]),r_resid_logn=r(e[ok],np.log(nn[ok])),w=w,frac_ok=ok.mean())
rows=[]
for fit,app in (('b10','b01'),('b01','b10')):
    P,G=E[app]['P'],E[app]['G']; f=fov[G]
    for j in c11:
        k=M[G,j]; p=P[k,j]; y=Y[G[k],j]; lm,nn=loo(p,f[k]); ok=~np.isnan(lm)
        w=res[(fit,names[j])]['w']; pn=p.copy(); pn[ok]=w[0]+w[1]*p[ok]+w[2]*lm[ok]+w[3]*np.log(nn[ok])
        # cells alone in FOV: keep affine of p only (scale-consistent): use w[1]*p + w[0]
        pn[~ok]=w[0]+w[1]*p[~ok]
        rows.append(dict(app=app,target=names[j],own=r(p,y),neigh=r(pn,y),r_resid_loo_app=res[(app,names[j])]['r_resid_loo'],r_resid_logn_app=res[(app,names[j])]['r_resid_logn'],w_loo_fit=w[2],w_logn_fit=w[3]))
df=pd.DataFrame(rows); pd.set_option('display.width',250); print(df.round(3).to_string(index=False))
g=df.groupby('app')[['own','neigh']].mean(); g.loc['GOAL']=g.mean(); print(g.round(4))
df.to_csv('/tmp/n095_ab/ab_neighbour_context.csv',index=False)
