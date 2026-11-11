"""N034 shared code: LF -> 6-channel fluorescence IMAGE generation (conditional flow matching / same-net regression baseline).
Target space (GPU): y = asinh(((x - bg) / gain) / a_c) / YS per channel, x = raw uint16 fluor, bg/gain/a_c from fm_prep.npz
(bg = border-ring median, gain = slide-position artefact, multiplicative; labels-only curation, LF inputs untouched).
Condition: normalised LF (23 LED + DPC = 24 ch, GPUNorm) + a 'condition present' channel (0 when dropped for CFG) + stain token
(which antibody the slide was stained with: CD123..HLA-DR, all, unstained); every cell is observed under exactly ONE stain."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from vs_common import W, RawSet, GPUNorm, amp, atomic_torch, rng_state, restore_rng, EpochBatches, make_loader, digest  # noqa: F401

YS = 3.0


class FluorSet(RawSet):
    """RawSet (LF raw int16, DPC, global row) + raw fluorescence image (6,128,128) as int32 (uint16 does not fit int16)."""
    def __init__(self, rows, pack=W + '/pack'):
        super().__init__(rows, pack=pack); self.fl = None

    def __getitem__(self, i):
        raw, ph, g = super().__getitem__(i)
        if self.fl is None:
            self.fl = np.load(self.pack + '/fluor.npy', mmap_mode='r')
        return raw, ph, g, torch.from_numpy(np.asarray(self.fl[g], np.int32))


class Prep:
    """GPU-side target transform + per-cell stain token from fm_prep.npz."""
    def __init__(self, device, path=W + '/fm_prep.npz'):
        P = np.load(path, allow_pickle=False)
        self.bg = torch.tensor(P['bg'], device=device); self.gain = torch.tensor(P['gain'], device=device)
        self.a = torch.tensor(P['a_c'], device=device)[None, :, None, None]
        self.stain = torch.tensor(P['stain'].astype(np.int64), device=device); self.stain_names = [str(s) for s in P['stain_names']]

    def target(self, fl, g):
        x = (fl.float() - self.bg[g][:, :, None, None]) / self.gain[g][:, :, None, None]
        return torch.asinh(x / self.a) / YS

    def linear(self, y):   # inverse: target space -> gain-corrected, background-subtracted counts
        return torch.sinh(y * YS) * self.a


def batch_to_device(batch, norm, prep, device, translate=0):
    raw, ph, g, fl = batch
    g = g.to(device, non_blocking=True)
    x = norm(raw.to(device, non_blocking=True), ph.to(device, non_blocking=True), g, ch='all', phase=True, shift=0)
    y = prep.target(fl.to(device, non_blocking=True), g)
    if translate:   # joint translation, replicate padding (LF and fluor share the 128 px pixel grid)
        r = translate; sh = torch.randint(-r, r + 1, (len(x), 2)).tolist()
        xp, yp = F.pad(x, (r, r, r, r), mode='replicate'), F.pad(y, (r, r, r, r), mode='replicate')
        x = torch.stack([xp[i, :, r - dy:r - dy + 128, r - dx:r - dx + 128] for i, (dy, dx) in enumerate(sh)])
        y = torch.stack([yp[i, :, r - dy:r - dy + 128, r - dx:r - dx + 128] for i, (dy, dx) in enumerate(sh)])
    return x.contiguous(), y.contiguous(), prep.stain[g], g


class TimeEmb(nn.Module):
    def __init__(self, d, de):
        super().__init__(); self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, de), nn.SiLU(), nn.Linear(de, de))

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000.) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.float()[:, None] * 1000. * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], 1))


class Res(nn.Module):
    def __init__(self, cin, cout, de, drop=0.):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(de, 2 * cout)
        self.n2, self.drop, self.c2 = nn.GroupNorm(32, cout), nn.Dropout(drop), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.e(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.c2(self.drop(F.silu(self.n2(h) * (1 + sc) + sh)))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__(); self.h = heads
        self.n, self.qkv, self.o = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)

    def forward(self, x):
        B, C, H, Wd = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * Wd).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v)            # (B, h, HW, d)
        return x + self.o(o.transpose(-1, -2).reshape(B, C, H, Wd))


class Blk(nn.Module):
    def __init__(self, cin, cout, de, attn, drop):
        super().__init__(); self.r = Res(cin, cout, de, drop); self.a = Attn(cout) if attn else None

    def forward(self, x, e):
        x = self.r(x, e)
        return self.a(x) if self.a is not None else x


class CondUNet(nn.Module):
    """x_t (6) ++ condition (24 LF + 1 present-flag) -> velocity (6). Time + stain embeddings drive AdaGN in every block."""
    def __init__(self, ch=(64, 128, 256, 256), nres=2, n_stain=10, attn_res=(16, 32), cin=6, ccond=25, drop=0.0, size=128, patch=1):
        super().__init__()
        self.patch = patch; size //= patch   # patch=2: pixel-unshuffle stem / pixel-shuffle head (128 px in/out, UNet runs at 64 px)
        de = 4 * ch[0]; self.temb = TimeEmb(ch[0], de); self.semb = nn.Embedding(n_stain, de)
        self.inp = nn.Conv2d((cin + ccond) * patch ** 2, ch[0], 3, padding=1)
        self.enc = nn.ModuleList(); skip = [ch[0]]; c = ch[0]; res = size
        for i, co in enumerate(ch):
            for _ in range(nres):
                self.enc.append(Blk(c, co, de, res in attn_res, drop)); c = co; skip.append(c)
            if i < len(ch) - 1:
                self.enc.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skip.append(c); res //= 2
        self.mid = nn.ModuleList([Blk(c, c, de, True, drop), Blk(c, c, de, False, drop)])
        self.dec = nn.ModuleList()
        for i in reversed(range(len(ch))):
            for _ in range(nres + 1):
                self.dec.append(Blk(c + skip.pop(), ch[i], de, res in attn_res, drop)); c = ch[i]
            if i > 0:
                self.dec.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c, c, 3, padding=1))); res *= 2
        assert not skip
        self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, cin * patch ** 2, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, xt, t, stain, cond):
        e = self.temb(t) + self.semb(stain)
        h = torch.cat([xt, cond], 1)
        h = self.inp(F.pixel_unshuffle(h, self.patch) if self.patch > 1 else h); hs = [h]
        for m in self.enc:
            h = m(h, e) if isinstance(m, Blk) else m(h); hs.append(h)
        for m in self.mid:
            h = m(h, e)
        for m in self.dec:
            h = m(torch.cat([h, hs.pop()], 1), e) if isinstance(m, Blk) else m(h)
        h = self.out(h)
        return F.pixel_shuffle(h, self.patch) if self.patch > 1 else h


def make_cond(x, drop_p=0.):
    """LF condition with a presence flag; drop_p>0 zeroes the LF for a random subset (classifier-free guidance training)."""
    keep = torch.ones(len(x), 1, 1, 1, device=x.device)
    if drop_p > 0:
        keep = (torch.rand(len(x), 1, 1, 1, device=x.device) >= drop_p).float()
    return torch.cat([x * keep, keep.expand(-1, 1, *x.shape[-2:])], 1)


@torch.no_grad()
def sample(net, x, stain, steps=16, cfg=1.0, generator=None):
    """Heun (2nd order) integration of dx/dt = v(x, t) from t=0 (noise) to t=1 (data); cfg>1 = classifier-free guidance on the LF."""
    cond = make_cond(x); null = torch.zeros_like(cond)          # null = LF zeroed AND presence flag 0 (as in CFG dropout)
    z = torch.randn(len(x), 6, *x.shape[-2:], device=x.device, generator=generator)
    def v(z, t):
        tt = torch.full((len(z),), float(t), device=z.device)
        with amp(z.device):
            u = net(z, tt, stain, cond).float()
            if cfg != 1.0:
                un = net(z, tt, stain, null).float(); u = un + cfg * (u - un)
        return u
    ts = np.linspace(0., 1., steps + 1)
    for i in range(steps):
        h = ts[i + 1] - ts[i]; d0 = v(z, ts[i]); zp = z + h * d0
        z = z + h * .5 * (d0 + v(zp, ts[i + 1]))
    return z


@torch.no_grad()
def regress(net, x, stain):
    with amp(x.device):
        return net(torch.zeros(len(x), 6, *x.shape[-2:], device=x.device), torch.zeros(len(x), device=x.device), stain, make_cond(x)).float()
