import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
def r(a,b): return np.corrcoef(a,b)[0,1]
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
E={d:np.load(f'/tmp/n095_ab/e8_{d}.npz') for d in ('b10','b01')}
def feats(P): 
    X=(P-P.mean(0))/P.std(0); return np.c_[np.ones(len(P)),X]
rows=[]
for fit,app in (('b10','b01'),('b01','b10')):
    Pf,Gf=E[fit]['P'],E[fit]['G']; Pa,Ga=E[app]['P'],E[app]['G']; Xf,Xa=feats(Pf),feats(Pa)
    for lam in (0.0,):
        for j in c11:
            kf=M[Gf,j]; ka=M[Ga,j]
            w=np.linalg.lstsq(Xf[kf],Y[Gf[kf],j],rcond=None)[0]
            base=r(Pa[ka,j],Y[Ga[ka],j]); st=r(Xa[ka]@w,Y[Ga[ka],j])
            # shrunk: half own head + half stacker (standardized)
            own=Xa[ka][:,1+j]; s=Xa[ka]@w; mix=r(own/own.std()+s/s.std(),Y[Ga[ka],j])
            rows.append(dict(fit_on=fit,applied_to=app,target=names[j],own=base,stacker=st,half_mix=mix))
df=pd.DataFrame(rows); print(df.round(3).to_string(index=False))
print(df.groupby('applied_to')[['own','stacker','half_mix']].mean().round(4))
df.to_csv('/tmp/n095_ab/ab_crossdir_stacker.csv',index=False)
