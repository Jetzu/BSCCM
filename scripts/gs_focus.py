"""GS (Wright) image QC: sharpness = mean gradient magnitude of the grey (sum RGB) image in the central 200x200 px (paper's histology
QC used summed gradients). Also mean optical density. Writes gs_qc.npz (hist-row order) into the workspace."""
import numpy as np, os
from multiprocessing import Pool
W='/hnvme/workspace/v181bb10-bsccm'
def one(r):
    h=np.load(W+'/pack/histology.npy',mmap_mode='r')[r].astype(np.float32).sum(2); c=h[99:299,99:299]
    gy,gx=np.gradient(c); bg=np.percentile(h,90)
    return float(np.hypot(gx,gy).mean()/bg), float(-np.log(np.clip(c/bg,1e-3,1.5)).mean())
if __name__=='__main__':
    n=np.load(W+'/pack/hist_cells.npy').shape[0]
    with Pool(48) as p: out=np.array(p.map(one,range(n),chunksize=64))
    np.savez(W+'/gs_qc.npz',sharp=out[:,0],od=out[:,1]); print('sharpness pct 1/10/50/90/99:',np.round(np.percentile(out[:,0],[1,10,50,90,99]),4))
