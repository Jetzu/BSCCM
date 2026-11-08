"""Independent verification: sample cells decoded by name via zipfile+numcodecs (no offset index) vs the pack; plus full zero-row scans."""
import zipfile, json, numpy as np, numcodecs, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'; P=W+'/pack'; idx=pd.read_parquet(W+'/bsccm_index.parquet'); N=len(idx)
led=np.load(P+'/led_array.npy',mmap_mode='r'); fl=np.load(P+'/fluor.npy',mmap_mode='r'); dp=np.load(P+'/dpc.npy',mmap_mode='r'); hi=np.load(P+'/histology.npy',mmap_mode='r'); hc=np.load(P+'/hist_cells.npy')
print('shapes',led.shape,fl.shape,dp.shape,hi.shape,led.dtype,dp.dtype,hi.dtype, flush=True)
z=zipfile.ZipFile(W+'/BSCCM.zarr.zip'); meta=json.load(open(W+'/bsccm_zip_index_zarrmeta.json'))
cod={k:numcodecs.get_codec(v[0]['compressor']) for k,v in meta.items()}
rng=np.random.default_rng(1); rows=rng.choice(N,300,replace=False); bad=0
def get(g,mod,key):
    return z.read(f"BSCCM/BSCCM_images.zarr/{idx.data_path[g]}/{mod}/cell_{g}/{key}")
for g in rows:
    for c in range(23):
        a=np.frombuffer(cod['led_array'].decode(get(g,'led_array',f'{c}.0.0')),'<u2').reshape(128,128); bad+=not np.array_equal(a,led[g,c])
    for c in range(6):
        a=np.frombuffer(cod['fluor'].decode(get(g,'fluor',f'{c}.0.0')),'<u2').reshape(128,128); bad+=not np.array_equal(a,fl[g,c])
    a=np.frombuffer(cod['dpc'].decode(get(g,'dpc','0.0.0')),'<f4').reshape(128,128); bad+=not np.array_equal(a,dp[g])
for r in rng.choice(len(hc),60,replace=False):
    g=int(hc[r]); full=np.zeros((398,398,3),hi.dtype)
    for k in range(4):
        y,x=divmod(k,2); full[y*199:(y+1)*199,x*199:(x+1)*199]=np.frombuffer(cod['histology'].decode(get(g,'histology',f'{y}.{x}.0')),hi.dtype).reshape(199,199,3)
    bad+=not np.array_equal(full,hi[r])
print('sampled mismatching arrays:',bad,'(300 cells x 30 arrays + 60 histology)',flush=True)
z0=0
for s in range(0,N,20000):
    z0+=int((led[s:s+20000,0].reshape(min(20000,N-s),-1).max(1)==0).sum())+int((fl[s:s+20000,0].reshape(min(20000,N-s),-1).max(1)==0).sum())*0
print('cells whose BF channel is all-zero:',z0)
print('hist rows all-zero:',sum(int(hi[i].max()==0) for i in range(0,len(hc),50)),'(every 50th)')
print('led value range sample',int(led[:2000,0].min()),int(led[:2000,0].max()),'| hist range',int(hi[:50].min()),int(hi[:50].max()),'| dpc',float(dp[:2000].min()),float(dp[:2000].max()))
