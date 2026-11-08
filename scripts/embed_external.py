"""N008: frozen DinoBloom embeddings of an external MGG smear set (PBC, Acevedo 2020) and of BSCCM Wright cells at MATCHED physical
scale, for the lineage-consistency anchor. Scale matching: median fractional cell area (Otsu on optical darkness, central component)
is measured in both sets; BSCCM crops are centre-cropped so that cell/crop area ratio equals PBC's, then both resized to 224.
Outputs (workspace emb/): pbc_dinobloom_<S>.npy + pbc_labels.npy (+ class names), giemsa_dinobloom_<S>_scalematched.npy, scale.json.
Runs on GPU if available, else CPU (48 threads).
usage: embed_external.py --size B"""
import argparse, io, json, os, zipfile, numpy as np, torch, torch.nn.functional as F, timm
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label
ap = argparse.ArgumentParser(); ap.add_argument('--size', default='B'); a = ap.parse_args()
W = '/hnvme/workspace/v181bb10-bsccm'; dev = 'cuda' if torch.cuda.is_available() else 'cpu'; torch.set_num_threads(48)
cfg = {'S': 'vit_small_patch14_dinov2', 'B': 'vit_base_patch14_dinov2', 'L': 'vit_large_patch14_dinov2'}[a.size]
m = timm.create_model(cfg, pretrained=False, num_classes=0, img_size=224)
sd = torch.load(f'{W}/models/DinoBloom-{a.size}.pth', map_location='cpu', weights_only=True)['teacher']
m.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith('backbone.') and k != 'backbone.mask_token'}, strict=True)
m = m.to(dev).eval(); dt = torch.float16 if dev == 'cuda' else torch.float32; m = m.to(dt)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def cell_frac(rgb01):   # fraction of crop covered by the central cell (darkness = 1 - mean RGB)
    d = 1 - rgb01.mean(2); msk = d > max(threshold_otsu(d), 0.08); L = label(msk); h, w = L.shape
    c = L[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8]; v = np.bincount(c.ravel(), minlength=L.max() + 1); v[0] = 0
    return float((L == v.argmax()).sum() / L.size) if v.max() > 0 else np.nan
@torch.no_grad()
def embed(batch01):     # list of HxWx3 float arrays in [0,1]
    x = torch.stack([F.interpolate(torch.tensor(b).permute(2, 0, 1)[None].float(), size=224, mode='bilinear', antialias=True)[0] for b in batch01])
    x = ((x - MEAN) / STD).to(dev, dt); return m(x).float().cpu().numpy()
# --- PBC
z = zipfile.ZipFile(f'{W}/external/pbc/PBC_dataset_normal_DIB.zip'); names = sorted(n for n in z.namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n and not os.path.basename(n).startswith('.'))   # drops macOS ._* forks and a stray .DS_* file (17,092 images = documented count)
classes = sorted({n.split('/')[-2] for n in names}); lab = np.array([classes.index(n.split('/')[-2]) for n in names])
E, fr, buf = [], [], []
for i, n in enumerate(names):
    im = np.asarray(Image.open(io.BytesIO(z.read(n))).convert('RGB'), np.float32) / 255.; buf.append(im)
    if i % 10 == 0: fr.append(cell_frac(im))
    if len(buf) == 256 or i == len(names) - 1: E.append(embed(buf)); buf = []
np.save(f'{W}/emb/pbc_dinobloom_{a.size}.npy', np.concatenate(E)); np.save(f'{W}/emb/pbc_labels.npy', lab); json.dump(classes, open(f'{W}/emb/pbc_classes.json', 'w'))
# --- BSCCM Wright, scale-matched centre crop
h = np.load(W + '/pack/histology.npy', mmap_mode='r'); sc = float(np.percentile(h[:500:5], 99.9))
frb = [cell_frac(np.clip(np.asarray(h[r], np.float32) / sc, 0, 1)) for r in range(0, len(h), 40)]
f_pbc, f_bs = float(np.nanmedian(fr)), float(np.nanmedian(frb)); side = int(round(398 * np.sqrt(f_bs / f_pbc))); side = min(398, max(160, side)); o = (398 - side) // 2
json.dump({'pbc_median_cell_frac': f_pbc, 'bsccm_median_cell_frac': f_bs, 'bsccm_crop_side_px': side, 'scale_ref': sc}, open(f'{W}/emb/scale.json', 'w'), indent=1)
print('scale match:', f_pbc, f_bs, 'crop side', side, flush=True)
E = []
for s in range(0, len(h), 256):
    E.append(embed([np.clip(np.asarray(h[r, o:o + side, o:o + side], np.float32) / sc, 0, 1) for r in range(s, min(s + 256, len(h)))]))
np.save(f'{W}/emb/giemsa_dinobloom_{a.size}_scalematched.npy', np.concatenate(E)); print('done', dev)
