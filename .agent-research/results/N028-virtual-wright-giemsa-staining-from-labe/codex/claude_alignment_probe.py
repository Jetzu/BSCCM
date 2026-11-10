import numpy as np, sys
sys.path.insert(0,'/home/hpc/v181bb/v181bb10/BSCCM')
from bsccm.data import chan_list
import torch, torch.nn.functional as F
W='/hnvme/workspace/v181bb10-bsccm'
H=np.load(W+'/pack/histology.npy',mmap_mode='r'); hc=np.load(W+'/pack/hist_cells.npy'); L=np.load(W+'/pack/led_array.npy',mmap_mode='r')
bf=chan_list('bf'); print('bf channels', bf)
rng=np.random.default_rng(0); idx=rng.choice(len(hc),200,replace=False)
def z(x): x=x-x.mean(); return x/(x.std()+1e-6)
res={}
for frac in [1.0,0.8,0.6,0.5,0.4,0.33,0.25]:
    peaks=[];shifts=[]
    for i in idx:
        g=int(hc[i]); lf=np.array(L[g]).view(np.int16)[bf].astype(np.float32).mean(0)
        gs=np.array(H[i]).astype(np.float32).mean(-1)
        s=int(398*frac); o=(398-s)//2; gsc=torch.tensor(gs[o:o+s,o:o+s])[None,None]
        gsr=F.interpolate(gsc,size=(128,128),mode='bilinear',antialias=True)[0,0].numpy()
        a=z(lf); b=z(gsr)
        c=np.real(np.fft.ifft2(np.fft.fft2(a)*np.conj(np.fft.fft2(b))))/a.size
        k=np.unravel_index(np.argmax(np.abs(c)),c.shape); peaks.append(c[k]); dy,dx=[(v if v<64 else v-128) for v in k]; shifts.append(np.hypot(dy,dx))
    peaks=np.array(peaks); res[frac]=(np.median(np.abs(peaks)), np.mean(np.sign(peaks)), np.median(shifts), np.percentile(shifts,90))
    print(f'GS centre fraction {frac:.2f}: median |peak corr| {res[frac][0]:.3f} sign mean {res[frac][1]:+.2f} median shift {res[frac][2]:.1f}px p90 {res[frac][3]:.1f}px')
