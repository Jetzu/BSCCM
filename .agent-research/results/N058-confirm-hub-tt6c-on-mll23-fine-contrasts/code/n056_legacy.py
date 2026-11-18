"""Verbatim legacy anchor loader and augmentation; generated from source AST spans."""
import os, numpy as np, torch
import torch.nn.functional as F
class ExtSet(torch.utils.data.Dataset):   # unlabelled external smear images (labels never read here)
    def __init__(s, zpath, half, anchors=None):
        import zipfile; s.zp = zpath; s.z = None
        names = sorted(n for n in zipfile.ZipFile(zpath).namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n and not os.path.basename(n).startswith('.'))
        s.names = names[0::2] if half == 'A' else names[1::2]
        if anchors: s.names = [n for n in s.names if n.split('/')[-2] in anchors]     # anchor classes only (eos/baso never seen)
        s.cls = [anchors.index(n.split('/')[-2]) if anchors and n.split('/')[-2] in anchors else -1 for n in s.names]
    def __len__(s): return len(s.names)
    def __getitem__(s, i):
        import zipfile, io
        from PIL import Image
        if s.z is None: s.z = zipfile.ZipFile(s.zp)
        im = Image.open(io.BytesIO(s.z.read(s.names[i]))).convert('RGB').resize((398, 398), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, np.uint8).copy()), s.cls[i]

def _macenko_stains(od, beta=0.15):
    # od: (3, P) optical density of one image; returns 3x2 stain matrix (unit columns) and 99th-pct concentrations
    m = od.sum(0) > beta; o = od[:, m] if m.sum() > 200 else od
    c = torch.cov(o); ev, V = torch.linalg.eigh(c); V = V[:, 1:]                       # top-2 eigenvectors (3x2)
    pr = V.T @ o; ang = torch.atan2(pr[1], pr[0]); lo, hi = torch.quantile(ang, 0.01), torch.quantile(ang, 0.99)
    v1 = V @ torch.stack([torch.cos(lo), torch.sin(lo)]); v2 = V @ torch.stack([torch.cos(hi), torch.sin(hi)])
    v1, v2 = v1 * torch.sign(v1.sum()), v2 * torch.sign(v2.sum())
    S = torch.stack([v1, v2], 1)
    S = S / S.norm(dim=0, keepdim=True); C = torch.linalg.lstsq(S, od).solution
    return S, torch.quantile(C[:, m] if m.sum() > 200 else C, 0.99, dim=1)

def macenko(x):   # x (B,3,H,W) in [0,1] -> stain-normalised to MAC_REF
    B, _, Hh, Ww = x.shape; out = torch.empty_like(x)
    for i in range(B):
        od = -torch.log(x[i].reshape(3, -1).clamp(min=1e-3)); S, cmax = _macenko_stains(od)
        # order the two stains consistently with the reference (match by cosine)
        if (S[:, 0] @ MAC_REF[0][:, 0]) < (S[:, 1] @ MAC_REF[0][:, 0]): S, cmax = S[:, [1, 0]], cmax[[1, 0]]
        C = torch.linalg.lstsq(S, od).solution * (MAC_REF[1] / cmax.clamp(min=1e-3))[:, None]
        out[i] = torch.exp(-(MAC_REF[0] @ C)).reshape(3, Hh, Ww)
    return out.clamp(0, 1)

def pipeline(x, train):   # x (B,3,398,398) float in [0,1] (BSCCM or resized external smears)
    if a.stain_norm == 'macenko': x = macenko(x)
    B = len(x)
    if train and a.aug:
        od = -torch.log(x.clamp(min=1e-3))                                                  # optical density
        od = od * torch.exp(0.25 * torch.randn(B, 3, 1, 1, device=dev))                     # per-channel stain intensity
        od = od * torch.exp(0.15 * torch.randn(B, 1, 1, 1, device=dev))                     # overall stain amount
        x = torch.exp(-od)
        x = x * (1 + 0.04 * torch.randn(B, 3, 1, 1, device=dev))                            # background tint / white balance
        g = torch.exp(0.2 * torch.randn(B, 1, 1, 1, device=dev)); x = x.clamp(1e-4, 1) ** g    # gamma
        sig = 2.0 * torch.rand(B, device=dev); k = torch.arange(-6, 7, device=dev).float()
        ker = torch.exp(-0.5 * (k[None] / sig.clamp(min=0.05)[:, None]) ** 2); ker = ker / ker.sum(1, keepdim=True)
        kk = ker[:, None, :].expand(B, 3, 13).reshape(B * 3, 1, 1, 13)
        y = F.conv2d(F.pad(x.reshape(1, B * 3, 398, 398), (6, 6, 0, 0), mode='reflect'), kk, groups=B * 3)
        x = F.conv2d(F.pad(y, (0, 0, 6, 6), mode='reflect'), kk.transpose(2, 3), groups=B * 3).reshape(B, 3, 398, 398)
        x = x + 0.02 * torch.randn_like(x)
        # smooth illumination / stain-thickness gradient across the crop (seen as slide-position artefact, N012)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 398, device=dev), torch.linspace(-1, 1, 398, device=dev), indexing='ij')
        gvec = 0.08 * torch.randn(B, 2, device=dev); x = x * (1 + gvec[:, 0, None, None, None] * xx + gvec[:, 1, None, None, None] * yy)
        # axial chromatic aberration: extra blur on ONE random colour channel (sequential R/G/B LED focus differences)
        c = int(torch.randint(3, (1,))); sg = float(1.5 * torch.rand(1)); 
        if sg > 0.2:
            kk1 = torch.exp(-0.5 * (k / sg) ** 2); kk1 = (kk1 / kk1.sum()).view(1, 1, 1, 13)
            xc = x[:, c:c + 1]; xc = F.conv2d(F.pad(xc, (6, 6, 0, 0), mode='reflect'), kk1); xc = F.conv2d(F.pad(xc, (0, 0, 6, 6), mode='reflect'), kk1.transpose(2, 3))
            x = torch.cat([x[:, :c], xc, x[:, c + 1:]], 1)
        s = (0.8 + 0.2 * torch.rand(B, device=dev)); t = (torch.rand(B, 2, device=dev) - 0.5) * 0.2 # resized crop: scale 0.8-1, shift
        th = torch.zeros(B, 2, 3, device=dev); th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, :, 2] = t
        x = F.grid_sample(x, F.affine_grid(th, (B, 3, 224, 224), align_corners=False), padding_mode='reflection', align_corners=False)
        k90 = int(torch.randint(4, (1,))); x = torch.rot90(x, k90, (2, 3))
        if torch.rand(1) < 0.5: x = x.flip(3)
    else:
        x = F.interpolate(x, size=224, mode='bilinear', antialias=True)
    return (x.clamp(0, 1) - MEAN) / STD
