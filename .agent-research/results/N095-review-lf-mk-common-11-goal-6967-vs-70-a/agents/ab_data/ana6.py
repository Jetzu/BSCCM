import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
def r(a,b): return np.corrcoef(a,b)[0,1]
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; b=T['batch']
c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
zi=np.load(f'{W}/runs/n039_ceil_fov_v5_mt0/eval_preds.npz'); Pi,Gi=zi['P'],zi['G']; posi={g:i for i,g in enumerate(Gi)}
print('indist keys',zi.files)
rows=[]
for d,draw in (('b10',0),('b01',1)):
    e=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=e['P'],e['G']
    C3=np.load(f'{W}/runs/n048_v9_base_b01/eval_preds.npz')['C3'] if d=='b01' else None
    ii=np.array([posi.get(g,-1) for g in G]); ok=ii>=0
    for j in c11:
        k=M[G,j]&ok; y=Y[G[k],j]; pe=P[k,j]; pi=Pi[ii[k],j]
        row=dict(dir=d,target=names[j],n=k.sum(),E8=r(pe,y),indist_v5trained=r(pi,y),blend=r(pe/pe.std()+pi/pi.std(),y))
        rows.append(row)
df=pd.DataFrame(rows); df['indist_minus_E8']=df.indist_v5trained-df.E8
print(df.round(3).to_string(index=False)); g=df.groupby('dir')[['E8','indist_v5trained','blend']].mean(); g.loc['GOAL']=g.mean(); print(g.round(4))
df.to_csv('/tmp/n095_ab/ab_indist_vs_e8.csv',index=False)
# class-level structure on b01 for key targets
e=np.load('/tmp/n095_ab/e8_b01.npz'); P,G=e['P'],e['G']; C3=np.load(f'{W}/runs/n048_v9_base_b01/eval_preds.npz')['C3']
ii=np.array([posi[g] for g in G])
for t in ('single:HLA-DR','grp:CD45','single:CD3','single:CD123','grp:CD16'):
    j=names.index(t); k=M[G,j]; y=Y[G[k],j]; pe=P[k,j]; pi=Pi[ii[k],j]; c=C3[k]
    print(t,'frac by class',[round((c==q).mean(),2) for q in range(3)])
    for q in range(3):
        m=c==q
        print(f'   cls{q}: y {y[m].mean():+.2f}±{y[m].std():.2f}  E8 {pe[m].mean():+.2f}±{pe[m].std():.2f} r {r(pe[m],y[m]):.2f} | indist {pi[m].mean():+.2f}±{pi[m].std():.2f} r {r(pi[m],y[m]):.2f}')
