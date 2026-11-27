import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; fov=T['fov']
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
rng=np.random.default_rng(0); B=200; out={}
for d in ('b10','b01'):
    e=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=e['P'],e['G']; f=fov[G]; u,inv=np.unique(f,return_inverse=True)
    idx_by=[np.where(inv==i)[0] for i in range(len(u))]
    vals=[]
    for b in range(B):
        pick=rng.integers(0,len(u),len(u)); rows=np.concatenate([idx_by[i] for i in pick])
        rs=[]
        for j in c11:
            k=rows[M[G[rows],j]]; rs.append(np.corrcoef(P[k,j],Y[G[k],j])[0,1])
        vals.append(np.mean(rs))
    out[d]=np.array(vals); print(d,'FOV-bootstrap mean %.4f sd %.4f'%(out[d].mean(),out[d].std()))
g=(out['b10']+out['b01'])/2; print('GOAL bootstrap sd %.4f; P(GOAL>=.70) %.2f'%(g.std(),(g>=.70).mean()))
np.savetxt('/tmp/n095_ab/ab_fov_bootstrap.csv',np.c_[out['b10'],out['b01'],g],delimiter=',',header='b10,b01,GOAL',comments='')
