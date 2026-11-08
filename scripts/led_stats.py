import numpy as np, sys
W='/hnvme/workspace/v181bb10-bsccm'; m=np.load(W+'/pack/led_array.npy',mmap_mode='r'); rng=np.random.default_rng(0)
rows=np.sort(rng.choice(m.shape[0],3000,replace=False)); a=np.log1p(np.stack([m[r].astype(np.float32) for r in rows]))
mu=a.mean((0,2,3)); sd=a.std((0,2,3)); np.savez(W+'/led_stats.npz',mu=mu,sd=sd); print(np.round(mu,2),np.round(sd,2))
