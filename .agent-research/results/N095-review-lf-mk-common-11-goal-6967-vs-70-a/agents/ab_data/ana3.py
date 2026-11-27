import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; sl=T['slide']
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
ix=names.index
def r(a,b): return np.corrcoef(a,b)[0,1]
pairs={'grp:CD45':['single:CD45'],'grp:CD16':['single:CD16'],'grp:CD3/CD19/CD56':['single:CD3','single:CD19','single:CD56'],
       'single:CD45':['grp:CD45'],'single:CD16':['grp:CD16']}
rows=[]
for d in ('b10','b01'):
    z=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=z['P'],z['G']
    base={names[j]:r(P[M[G,j],j],Y[G[M[G,j]],j]) for j in c11}
    for t,srcs in pairs.items():
        j=ix(t); k=M[G,j]; y=Y[G[k],j]
        own=P[k,j]; other=np.sum([P[k,ix(s)] for s in srcs],0)
        for w in (0,.25,.5,.75,1):
            rows.append(dict(dir=d,target=t,w_other=w,r=r((1-w)*own+w*other,y)))
    # also 3 pooled-group blend at w=.5 -> GOAL
    Pn=P.copy()
    for t,srcs in pairs.items():
        if not t.startswith('grp'): continue
        j=ix(t); Pn[:,j]=.5*P[:,j]+.5*np.sum([P[:,ix(s)] for s in srcs],0)
    sc={names[j]:r(Pn[M[G,j],j],Y[G[M[G,j]],j]) for j in c11}
    print(d,'base',round(np.mean(list(base.values())),4),'pooled-blend w.5',round(np.mean(list(sc.values())),4))
df=pd.DataFrame(rows); print(df.pivot_table(index=['target','w_other'],columns='dir',values='r').round(3).to_string())
df.to_csv('/tmp/n095_ab/ab_crosshead.csv',index=False)
# replicate-wise for grp:CD45 b01 with single head
z=np.load('/tmp/n095_ab/e8_b01.npz'); P,G=z['P'],z['G']
for s in (14,13):
    j=ix('grp:CD45'); k=M[G,j]&(sl[G]==s); y=Y[G[k],j]
    print('rep slide',s,'grp head',round(r(P[k,j],y),3),'single:CD45 head',round(r(P[k,ix('single:CD45')],y),3))
