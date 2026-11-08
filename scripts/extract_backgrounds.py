"""Extract the 50th-percentile flat-field backgrounds (one per LED channel, 2056x2056) -> backgrounds_p50.npy (23,2056,2056) f4, LED order."""
import zarr, numpy as np, sys
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM'); from bsccm.data import LED
W='/hnvme/workspace/v181bb10-bsccm'
s=zarr.storage.ZipStore(W+'/BSCCM.zarr.zip',mode='r'); g=zarr.open_group(s,mode='r',path='BSCCM/BSCCM_backgrounds.zarr')
out=np.stack([np.asarray(g[c]['50_percentile'][:],dtype=np.float32) for c in LED])
np.save(W+'/backgrounds_p50.npy',out); print(out.shape, [round(float(np.median(o)),1) for o in out])
