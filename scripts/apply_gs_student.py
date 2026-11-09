"""N019: apply a trained GS student (train_gs_bind.py output gs_bind.pt) to Wright/Giemsa RGB images and test lineage consistency.
Sources: --bsccm (all 15,712 BSCCM Wright cells, same preprocessing as training) and --pbc (Acevedo PBC smears, 8 expert classes, CC BY 4.0;
images read straight from the zip, [0,1] RGB -> 224). Writes per-cell student predictions (13 targets, z-units) + embeddings, and for PBC the
per-class mean predictions and AUROCs of pre-registered lineage contrasts (stated before looking at results, from the hematology review N002-D):
  CD16: neutrophil > lymphocyte, neutrophil > eosinophil (eosinophils are CD16-negative)
  CD14: monocyte > (neutrophil, lymphocyte)        grp CD3/CD19/CD56 and single CD3: lymphocyte > (neutrophil, monocyte)
  CD45: lymphocyte > neutrophil (CD45-bright lymphocytes)      CD123: basophil > others      HLA-DR: monocyte > neutrophil
usage: apply_gs_student.py --ckpt run/gs_bind.pt --lf-dim 512 --out DIR [--tta 1]"""
import argparse, io, json, os, zipfile, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--lf-dim', type=int, default=512); ap.add_argument('--dim', type=int, default=256)
ap.add_argument('--out', required=True); ap.add_argument('--tta', type=int, default=1); ap.add_argument('--pbc-half', default='all'); ap.add_argument('--pbc-bg', default='keep', choices=['keep', 'white']); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
W = '/hnvme/workspace/v181bb10-bsccm'; dev = 'cuda'; T = np.load(W + '/targets_v4.npz'); names = list(T['names']); K = len(names)
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, o))
vit = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=0, img_size=224)
net = nn.ModuleDict({'vit': vit, 'PG': mlp(768, a.dim), 'S': mlp(a.dim, K)})
sd = torch.load(a.ckpt, map_location='cpu', weights_only=True); sd = {k: v for k, v in sd.items() if k.split('.')[0] in ('vit', 'PG', 'S')}
print(net.load_state_dict(sd, strict=True)); net = net.to(dev).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def predict(batch01):   # list of HxWx3 [0,1] arrays -> (preds (B,K), z (B,dim)); TTA = mean over rot90 x flip (Wright RGB is isotropic)
    x = torch.stack([F.interpolate(torch.tensor(np.ascontiguousarray(b), dtype=torch.float32).permute(2, 0, 1)[None], size=224, mode='bilinear', antialias=True)[0] for b in batch01]).to(dev)
    x = (x.clamp(0, 1) - MEAN) / STD; P, Z = 0, 0; views = [(k, f) for k in range(4) for f in (0, 1)] if a.tta else [(0, 0)]
    for k, f in views:
        xv = torch.rot90(x, k, (2, 3)); xv = xv.flip(3) if f else xv
        with torch.autocast('cuda', torch.bfloat16): z = F.normalize(net['PG'](net['vit'](xv).float()), dim=-1)
        P = P + net['S'](z * math.sqrt(a.dim)).float(); Z = Z + z
    return (P / len(views)).cpu().numpy(), F.normalize(Z, dim=-1).cpu().numpy()
# --- BSCCM Wright cells
H = np.load(W + '/pack/histology.npy', mmap_mode='r'); scale = float(np.percentile(H[:500:5], 99.9)); Pb, Zb = [], []
for s in range(0, len(H), 256):
    p, z = predict([np.clip(np.asarray(H[r], np.float32) / scale, 0, 1) for r in range(s, min(s + 256, len(H)))]); Pb.append(p); Zb.append(z)
np.savez(a.out + '/bsccm_gs_student.npz', pred=np.concatenate(Pb), z=np.concatenate(Zb), names=np.array(names))
def whiten_bg(im):   # keep the central white blood cell, replace RBC/plasma background by BSCCM-like white (test-time only)
    from scipy import ndimage
    r, g, b_ = im[..., 0], im[..., 1], im[..., 2]; dark = 1 - im.mean(2); blue = b_ - r
    m = (blue > 0.02) | (dark > 0.45)                          # WBC nucleus/cytoplasm: bluish or dark; RBCs: reddish-pink
    m = ndimage.binary_closing(m, iterations=3); m = ndimage.binary_fill_holes(m); lab, n = ndimage.label(m)
    if n == 0: return im
    h, w = m.shape; c = lab[h // 2 - 15:h // 2 + 15, w // 2 - 15:w // 2 + 15]; v = np.bincount(c.ravel(), minlength=n + 1); v[0] = 0
    k = v.argmax() if v.max() > 0 else np.bincount(lab.ravel())[1:].argmax() + 1
    keep = ndimage.binary_dilation(lab == k, iterations=14); soft = ndimage.gaussian_filter(keep.astype(np.float32), 3)[..., None]   # generous margin keeps pale cytoplasm/granules
    return im * soft + np.array([0.98, 0.965, 0.96], np.float32) * (1 - soft)
# --- PBC smears
zf = zipfile.ZipFile(W + '/external/pbc/PBC_dataset_normal_DIB.zip')
fn = sorted(n for n in zf.namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n and not os.path.basename(n).startswith('.'))
fn = fn if a.pbc_half == 'all' else (fn[0::2] if a.pbc_half == 'A' else fn[1::2])   # half B = never used for adaptation
cls = np.array([n.split('/')[-2] for n in fn]); Pp, Zp, buf = [], [], []
for i, n in enumerate(fn):
    im = np.asarray(Image.open(io.BytesIO(zf.read(n))).convert('RGB'), np.float32) / 255.
    buf.append(whiten_bg(im) if a.pbc_bg == 'white' else im)
    if len(buf) == 256 or i == len(fn) - 1: p, z = predict(buf); Pp.append(p); Zp.append(z); buf = []
Pp = np.concatenate(Pp); np.savez(a.out + '/pbc_gs_student.npz', pred=Pp, z=np.concatenate(Zp), cls=cls, names=np.array(names))
res = {'pbc_class_means': {c: {names[k]: round(float(Pp[cls == c, k].mean()), 3) for k in range(K)} for c in sorted(set(cls))}}
def auc(marker, pos, neg):
    k = names.index(marker); m = np.isin(cls, pos + neg); return float(roc_auc_score(np.isin(cls[m], pos), Pp[m, k]))
C = {'CD16 neutrophil>lymphocyte': ('single:CD16', ['neutrophil'], ['lymphocyte']), 'CD16 neutrophil>eosinophil': ('single:CD16', ['neutrophil'], ['eosinophil']),
     'CD14 monocyte>neutrophil+lymphocyte': ('single:CD14', ['monocyte'], ['neutrophil', 'lymphocyte']),
     'grpCD3/19/56 lymphocyte>neutrophil+monocyte': ('grp:CD3/CD19/CD56', ['lymphocyte'], ['neutrophil', 'monocyte']),
     'CD3 lymphocyte>neutrophil+monocyte': ('single:CD3', ['lymphocyte'], ['neutrophil', 'monocyte']), 'CD45 lymphocyte>neutrophil': ('single:CD45', ['lymphocyte'], ['neutrophil']),
     'CD123 basophil>rest': ('single:CD123', ['basophil'], ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil']), 'HLA-DR monocyte>neutrophil': ('single:HLA-DR', ['monocyte'], ['neutrophil'])}
res['pbc_lineage_auroc'] = {k: round(auc(*v), 3) for k, v in C.items()}
json.dump(res, open(a.out + '/lineage_consistency.json', 'w'), indent=1); print(json.dumps(res['pbc_lineage_auroc'], indent=1))
