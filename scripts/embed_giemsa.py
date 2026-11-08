"""Frozen DinoBloom embeddings (CLS) of Giemsa/Wright histology crops. usage: embed_giemsa.py <S|B|L> <out.npy>
Images (398x398x3 uint16) -> float, scaled by the 99.9th percentile of the first 500 images, resized 224, ImageNet norm."""
import sys, numpy as np, torch, torch.nn.functional as F
W='/hnvme/workspace/v181bb10-bsccm'; size=sys.argv[1]; out=sys.argv[2]
cfg={'S':('vit_small_patch14_dinov2',384),'B':('vit_base_patch14_dinov2',768),'L':('vit_large_patch14_dinov2',1024)}[size]
import timm
m=timm.create_model(cfg[0],pretrained=False,num_classes=0,img_size=224,dynamic_img_size=True)
sd=torch.load(f'{W}/models/DinoBloom-{size}.pth',map_location='cpu',weights_only=True)['teacher']
sd={k[len('backbone.'):]:v for k,v in sd.items() if k.startswith('backbone.') and k!='backbone.mask_token'}
r=m.load_state_dict(sd,strict=True); print('load ok',r); m=m.cuda().eval().half()
h=np.load(W+'/pack/histology.npy',mmap_mode='r'); n=len(h)
sc=float(np.percentile(h[:500:5],99.9)); print('scale',sc,'dtype',h.dtype)
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda(); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
E=np.zeros((n,cfg[1]),np.float32)
with torch.no_grad():
    for s in range(0,n,128):
        x=torch.from_numpy(np.ascontiguousarray(h[s:s+128]).astype(np.float32)/sc).clamp(0,1).permute(0,3,1,2).cuda()
        x=F.interpolate(x,size=224,mode='bilinear',antialias=True); x=((x-mean)/std).half()
        E[s:s+128]=m(x).float().cpu().numpy()
np.save(out,E); print('saved',E.shape,'finite',np.isfinite(E).all(),'std',E.std())
