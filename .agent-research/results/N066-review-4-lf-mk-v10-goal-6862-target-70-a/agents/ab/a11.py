import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); M,C,b=T['mask'],T['censored'],T['batch']; names=[str(n) for n in T['names']]
print({names[j]:tuple(round(float(C[M[:,j]&(b==bb),j].mean()),3) for bb in (0,1)) for j in range(13)})
