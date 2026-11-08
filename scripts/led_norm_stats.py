"""N005: LED flat-field normalisation stats. ratio = (raw-30)/(bg_crop-30), bg = global 50th-pct flat field cropped at the cell's FOV position
(edge-padded). Per slide x channel scale = median of the ratio in the crop's outer 6-px ring (background). Final feature: log(ratio/slide_scale)/s_c.
Check: spread of background-ring level across FOV positions before vs after flat-fielding."""
import numpy as np, pandas as pd, json, sys
W='/hnvme/workspace/v181bb10-bsccm'; OUT=sys.argv[1]
idx=pd.read_parquet(W+'/bsccm_index.parquet'); led=np.load(W+'/pack/led_array.npy',mmap_mode='r'); bg=np.load(W+'/backgrounds_p50.npy',mmap_mode='r')
py=idx.position_in_fov_y_pix.values.astype(int); px=idx.position_in_fov_x_pix.values.astype(int); slide,snames=pd.factorize(idx.data_path)
np.savez(W+'/cellmeta.npz',pos_y=py,pos_x=px,slide=slide.astype(np.int16),slide_names=np.array(snames))
def bgcrop(g):
    y0,x0=py[g]-64,px[g]-64; H=bg.shape[1]
    ys=np.clip(np.arange(y0,y0+128),0,H-1); xs=np.clip(np.arange(x0,x0+128),0,H-1)
    return np.asarray(bg[:,ys][:,:,xs])
ring=np.zeros((128,128),bool); ring[:6]=ring[-6:]=True; ring[:,:6]=ring[:,-6:]=True
rng=np.random.default_rng(0); S=len(snames); scale=np.zeros((S,23),np.float32); raw_lvl=[]; ff_lvl=[]; pos=[]; feats=[]
for s in range(S):
    rows=rng.choice(np.where(slide==s)[0],300,replace=False)
    r=[]
    for g in rows:
        x=led[g].astype(np.float32); b=bgcrop(g); rat=(x-30)/(b-30)
        r.append(np.median(rat[:,ring],axis=1)); raw_lvl.append(np.median(x[:,ring],axis=1)); ff_lvl.append(r[-1]); pos.append(np.hypot(py[g]-1028,px[g]-1028))
    scale[s]=np.median(np.stack(r),0)
raw_lvl=np.stack(raw_lvl); ff_lvl=np.stack(ff_lvl); pos=np.array(pos)
# feature scale s_c on 2000 random cells
for g in rng.choice(len(idx),2000,replace=False):
    x=led[g].astype(np.float32); b=bgcrop(g); feats.append(np.log(np.clip((x-30)/(b-30)/scale[slide[g]][:,None,None],1e-3,None)).std(axis=(1,2)))
s_c=np.median(np.stack(feats),0)
np.savez(W+'/led_norm.npz',slide_scale=scale,s_c=s_c,slide_names=np.array(snames))
from scipy.stats import spearmanr
res={'radial_corr_raw_ring':[round(float(spearmanr(pos,raw_lvl[:,c]).correlation),3) for c in range(23)],
     'radial_corr_flatfielded_ring':[round(float(spearmanr(pos,ff_lvl[:,c]).correlation),3) for c in range(23)],
     'cv_ring_raw':[round(float(raw_lvl[:,c].std()/raw_lvl[:,c].mean()),3) for c in range(23)],
     'cv_ring_ff_after_slide_scale':[round(float((ff_lvl[:,c]/np.repeat(scale[:,c],300)).std()),3) for c in range(23)],
     'slide_scale_range':[[round(float(scale[:,c].min()),3),round(float(scale[:,c].max()),3)] for c in range(23)],'s_c':[round(float(v),3) for v in s_c]}
json.dump(res,open(OUT+'/led_norm_check.json','w'),indent=0); print(json.dumps(res))
