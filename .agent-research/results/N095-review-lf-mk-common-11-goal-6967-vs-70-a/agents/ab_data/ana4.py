import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; sl,fov=T['slide'],T['fov']
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def r(a,b): return np.corrcoef(a,b)[0,1]
def gmean(x,g):
    u,inv=np.unique(g,return_inverse=True); return (np.bincount(inv,x)/np.bincount(inv))[inv]
rng=np.random.default_rng(0); rows=[]
for d in ('b10','b01'):
    z=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=z['P'],z['G']
    C3=np.load(f'{W}/runs/n048_v9_base_{d}/eval_preds.npz')['C3']; assert np.array_equal(np.load(f'{W}/runs/n048_v9_base_{d}/eval_preds.npz')['G'],G)
    for j in c11:
        k=M[G,j]; p=P[k,j]; y=Y[G[k],j]; f=fov[G[k]]; s=sl[G[k]]; c=C3[k]
        def lab_fov_rm(ff): return r(p, y-gmean(y,ff)+gmean(p,ff))
        def wfov(ff): return r(p-gmean(p,ff), y-gmean(y,ff))
        real_rm, real_w = lab_fov_rm(f), wfov(f)
        nul_rm=[];nul_w=[]
        for it in range(20):
            ff=f.copy()
            for u in np.unique(s):
                m=np.where(s==u)[0]; ff[m]=f[m][rng.permutation(len(m))]
            nul_rm.append(lab_fov_rm(ff)); nul_w.append(wfov(ff))
        # type decomposition with predicted 3-class
        cm=gmean(y,c)  # type-oracle: test-label class means (oracle)
        r_type=r(cm,y); wt=np.mean([r(p[c==q],y[c==q]) for q in range(3) if (c==q).sum()>30])
        # share of label variance between predicted types
        between=np.var(cm)/np.var(y)
        rows.append(dict(dir=d,target=names[j],r=r(p,y),labfov_rm=real_rm,labfov_rm_null=np.mean(nul_rm),within_fov=real_w,within_fov_null=np.mean(nul_w),
                         type_oracle=r_type,within_type_r_mean=wt,between_type_var_share=between))
df=pd.DataFrame(rows); df['labfov_excess']=df.labfov_rm-df.labfov_rm_null; df['wfov_excess']=df.within_fov-df.within_fov_null
pd.set_option('display.width',250); print(df.round(3).to_string(index=False))
g=df.groupby('dir')[['r','labfov_rm','labfov_rm_null','labfov_excess','within_fov','within_fov_null','wfov_excess','type_oracle','within_type_r_mean','between_type_var_share']].mean()
g.loc['GOAL']=g.mean(); print(g.round(4).to_string())
df.to_csv('/tmp/n095_ab/ab_fov_null_type.csv',index=False); g.to_csv('/tmp/n095_ab/ab_fov_null_type_goal.csv')
