"""Illumination-preserving views and SIGReg for label-free SSL.

rng is a torch.Generator on the input device (CPU in DataLoader workers).
No spatial operation permutes illumination directions.
"""
import math

import torch
import torch.nn.functional as F


def _augment(x, rng):
    def rand(*shape):
        return torch.rand(shape, generator=rng, device=x.device)

    # Positive isotropic inverse scale and translation only: never rotate/flip.
    scale = 0.9 + 0.2 * rand().item()
    dx, dy = (12 * rand(2) - 6).tolist()
    # Divide translation by scale too: output displacement must stay <=6 px.
    theta = x.new_tensor([[1 / scale, 0, -2 * dx / (x.shape[2] * scale)],
                          [0, 1 / scale, -2 * dy / (x.shape[1] * scale)]])
    grid = F.affine_grid(theta[None], (1, *x.shape), align_corners=False)
    x = F.grid_sample(x[None], grid, padding_mode='reflection',
                      align_corners=False)[0]
    sigma = 0.8 * rand().item()
    if sigma > 1e-3:
        radius = math.ceil(3 * sigma)
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-0.5 * (t / sigma).square())
        k = k / k.sum()
        c = x.shape[0]
        x = F.conv2d(F.pad(x[None], (radius, radius, 0, 0), mode='reflect'),
                     k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'),
                     k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)[0]
    gain = torch.randn((24, 1, 1), generator=rng, device=x.device) * 0.05
    noise = torch.randn(x.shape, generator=rng, device=x.device) * 0.05
    return x * gain.exp() + noise


def make_views(x, arm, n_views, rng):
    """Return V tensors [24 augmented data channels, 24 presence planes]."""
    if x.shape != (24, 128, 128) or x.dtype != torch.float32:
        raise ValueError('expected float32 (24,128,128) LFNorm input')
    if arm not in ('aug', 'xillum') or n_views < 2:
        raise ValueError('arm must be aug|xillum; n_views must be >= 2')
    if arm == 'xillum' and n_views > 6:
        raise ValueError('24 channels allow at most 6 disjoint views of >=4 channels')
    if arm == 'xillum':
        groups = torch.tensor_split(torch.randperm(24, generator=rng, device=x.device), n_views)
    views = []
    for v in range(n_views):
        if arm == 'aug':
            keep = torch.rand(24, generator=rng, device=x.device) >= 0.1
            if not keep.any():
                keep[torch.randint(24, (), generator=rng, device=x.device)] = True
        else:
            keep = torch.zeros(24, dtype=torch.bool, device=x.device)
            keep[groups[v]] = True
        mask = keep.to(x.dtype)[:, None, None].expand_as(x)
        data = _augment(x, rng).masked_fill(~keep[:, None, None], 0)
        views.append(torch.cat((data, mask), dim=0))
    return views


def sigreg(z, n_dirs=256, n_t=17, t_max=5.0, scale_by_batch=True):
    """Weighted Epps–Pulley CF discrepancy against N(0,1), without whitening.

    Random directions are resampled on every call. Float32 arithmetic for bf16
    embeddings; double precision inputs retain double precision.
    """
    if z.ndim != 2 or min(z.shape) < 1 or n_dirs < 1 or n_t < 2 or t_max <= 0:
        raise ValueError('invalid embedding shape or integration parameters')
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z if z.dtype == torch.float64 else z.float()
        a = torch.randn(z.shape[1], n_dirs, device=z.device, dtype=z.dtype)
        a = F.normalize(a, dim=0)
        p = z @ a
        t = torch.linspace(-t_max, t_max, n_t, device=z.device, dtype=z.dtype)
        phase = p[:, :, None] * t
        real, imag = phase.cos().mean(0), phase.sin().mean(0)
        target = (-t.square() / 2).exp()
        integrand = ((real - target).square() + imag.square()) * target
        stat = torch.trapezoid(integrand, t, dim=-1).mean()
        return stat * z.shape[0] if scale_by_batch else stat


def ssl_loss(zs, lam=0.05):
    """Return (total, prediction squared L2 norm, SIGReg), all differentiable."""
    if len(zs) < 2 or not 0 <= lam <= 1:
        raise ValueError('need >=2 views and lam in [0,1]')
    if any(z.shape != zs[0].shape for z in zs):
        raise ValueError('view embeddings must have identical shapes')
    z = torch.stack([v.float() for v in zs])
    centre = z.mean(0)
    pred = (z - centre).square().sum(-1).mean()
    reg = torch.stack([sigreg(v) for v in zs]).mean()
    return (1 - lam) * pred + lam * reg, pred, reg


def make_views_batch(x, arm, n_views, gen):
    """GPU-batched version of make_views (same view semantics; integrator addition, N007).
    x: (B,24,H,W) float32 normalised LF (+phase) on GPU; gen: torch.Generator on x.device.
    Returns list of V tensors (B,48,H,W) = [augmented channels with dropped ones zeroed, presence planes]."""
    B, C, H, Wd = x.shape
    if C != 24: raise ValueError('expected 24 channels')
    if arm not in ('aug', 'xillum') or n_views < 2: raise ValueError('arm aug|xillum, n_views>=2')
    dev = x.device
    if arm == 'xillum':   # per-sample random balanced partition of the 24 channels into V disjoint groups
        rank = torch.rand(B, C, generator=gen, device=dev).argsort(1).argsort(1)            # random rank 0..23 per channel
        group = (rank * n_views) // C                                                       # balanced: 24/V channels per view
    views = []
    for v in range(n_views):
        if arm == 'aug':
            keep = torch.rand(B, C, generator=gen, device=dev) >= 0.1
            none = ~keep.any(1)
            if none.any(): keep[none, torch.randint(C, (int(none.sum()),), generator=gen, device=dev)] = True
        else:
            keep = group == v
        # geometry: isotropic scale 0.9-1.1 + translation +-6 px (output px), reflection padding; no rotation/flip
        sc = 0.9 + 0.2 * torch.rand(B, generator=gen, device=dev); d = 12 * torch.rand(B, 2, generator=gen, device=dev) - 6
        theta = torch.zeros(B, 2, 3, device=dev); theta[:, 0, 0] = 1 / sc; theta[:, 1, 1] = 1 / sc
        theta[:, 0, 2] = -2 * d[:, 0] / (Wd * sc); theta[:, 1, 2] = -2 * d[:, 1] / (H * sc)
        y = F.grid_sample(x, F.affine_grid(theta, (B, C, H, Wd), align_corners=False), padding_mode='reflection', align_corners=False)
        # per-sample Gaussian blur, sigma U(0,0.8), separable depthwise conv with per-sample kernels (radius 3)
        sig = (0.8 * torch.rand(B, generator=gen, device=dev)).clamp(min=1e-3); t = torch.arange(-3, 4, device=dev, dtype=x.dtype)
        k = torch.exp(-0.5 * (t[None] / sig[:, None]) ** 2); k = k / k.sum(1, keepdim=True)                # (B,7)
        kk = k[:, None, :].expand(B, C, 7).reshape(B * C, 1, 1, 7)
        y = F.conv2d(F.pad(y.reshape(1, B * C, H, Wd), (3, 3, 0, 0), mode='reflect'), kk, groups=B * C)
        y = F.conv2d(F.pad(y, (0, 0, 3, 3), mode='reflect'), kk.transpose(2, 3), groups=B * C).reshape(B, C, H, Wd)
        y = y * torch.exp(0.05 * torch.randn(B, C, 1, 1, generator=gen, device=dev)) + 0.05 * torch.randn(y.shape, generator=gen, device=dev)
        m = keep.to(x.dtype)[:, :, None, None]
        views.append(torch.cat([y * m, m.expand(B, C, H, Wd)], 1))
    return views
