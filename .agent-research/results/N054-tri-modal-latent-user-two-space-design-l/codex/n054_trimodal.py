#!/usr/bin/env python3
"""N054: frozen LF / masked marker matrix / image-level GS experiments.
All imports are package dependencies or local siblings; no repository path imports.
Run --help and read README.md for protocol and checkpoint contracts.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = '/hnvme/workspace/v181bb10-bsccm'
STOP = False

def stop_handler(signum, frame):
    global STOP
    STOP = True
    print(f'signal {signum}: checkpoint and exit at next optimizer boundary', flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def json_write(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, Path): return str(x)
        return x
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n'); os.replace(tmp, path)


def csv_write(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def corr(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-10 or y.std() < 1e-10: return None
    return float(np.corrcoef(x, y)[0, 1])


def product(a, b): return None if a is None or b is None else a * b


def mlp(i, o, hidden=512, depth=2):
    layers = []
    for _ in range(depth): layers += [nn.Linear(i, hidden), nn.GELU(), nn.LayerNorm(hidden)]; i = hidden
    return nn.Sequential(*layers, nn.Linear(i, o))


def legacy_mlp(i, o):
    return nn.Sequential(nn.Linear(i, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, o))


class SigLIP(nn.Module):
    def __init__(self):
        super().__init__(); self.t = nn.Parameter(torch.tensor(math.log(10.))); self.b = nn.Parameter(torch.tensor(-10.))
    def forward(self, a, b, ignore=None):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        logits = a @ b.T * self.t.exp().clamp(max=100) + self.b
        labels = 2 * torch.eye(len(a), device=a.device) - 1
        loss = -F.logsigmoid(labels * logits)
        if ignore is not None: loss = loss.masked_fill(ignore, 0)
        return loss.sum() / len(a)


def near_duplicate(y, m, tau=.25):
    # Verbatim criterion from train_bind.py: identical observed surface set,
    # all common readouts within tau; autofl excluded; diagonal remains positive.
    y, m = y[:, :12], m[:, :12]
    same = (m[:, None] == m[None]).all(-1) & (m.sum(1)[:, None] > 0)
    diff = ((y[:, None] - y[None]).abs() * m[:, None]).amax(-1)
    ignore = same & (diff < tau); ignore.fill_diagonal_(False)
    return ignore


def mse_mask(pred, y, m):
    # Equal target weighting (prevents large single-marker slides dominating).
    num = ((pred - y).square() * m).sum(0); den = m.sum(0)
    return (num / den.clamp(min=1))[den > 0].mean() if (den > 0).any() else pred.sum() * 0


class Spaces(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.P = mlp(1024, dim); self.EM = mlp(26, dim)
        self.H = mlp(dim, 13); self.sig = SigLIP(); self.dim = dim
    def lf(self, x): return F.normalize(self.P(x), dim=-1)
    def marker(self, y, m): return F.normalize(self.EM(torch.cat([y * m, m.float()], -1)), dim=-1)
    def decode(self, z): return self.H(z * math.sqrt(self.dim))
    def forward(self, x): return self.decode(self.lf(x))


class Velocity(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.net = mlp(dim + 16, dim)
        self.register_buffer('freq', 2 ** torch.arange(8).float() * math.pi)
    def forward(self, x, t):
        t = t.reshape(-1, 1) * self.freq[None]
        return self.net(torch.cat([x, t.sin(), t.cos()], -1))


@torch.no_grad()
def integrate(flow, z, steps=32):
    # Endpoints live on the unit sphere. The stochastic training bridge has zero
    # endpoint noise; inference uses deterministic probability-flow Euler ODE.
    x = z.clone()
    for k in range(steps): x = x + flow(x, x.new_full((len(x),), k / steps)) / steps
    return F.normalize(x, dim=-1)


class Data:
    def __init__(self, a):
        self.a = a; w = Path(a.data)
        with np.load(w / a.targets) as t:
            self.y = t['Y'].astype(np.float32); self.m = t['mask'].astype(bool)
            self.split, self.batch, self.hist, self.fov, self.slide = [t[k].copy() for k in ('split','batch','hist_row','fov','slide')]
            self.names = [str(n) for n in t['names']]
        assert len(self.names) == 13 and self.names[-1] == 'autofl'
        self.raw_m = self.m.copy()
        bad = (self.batch == 1) & self.m[:, 8]
        self.m[bad, 8] = False  # K019, not implemented by the supplied tt6c file.
        self.y = np.where(self.m, self.y, 0).astype(np.float32)
        assert np.isfinite(self.y).all()
        self.features, self.hist_features, self.hlookup = [], [], []
        for d in (a.lf_marker, a.lf_autofl):
            d = Path(d); d = d if d.is_absolute() else w / d
            x = np.load(d / 'lf_feats_all.npy', mmap_mode='r'); assert x.shape == (len(self.y), 512)
            h = np.load(d / 'lf_feats_hist.npy', mmap_mode='r')
            indices = np.load(d / 'lf_feats_hist_indices.npy'); hr = np.load(d / 'lf_feats_hist_hist_row.npy')
            assert np.array_equal(self.hist[indices], hr) and len(np.unique(hr)) == len(hr)
            lut = np.full(len(self.y), -1, np.int64); lut[indices] = np.arange(len(indices))
            assert (lut[self.hist >= 0] >= 0).all()
            # hist arrays explicitly carry own-cell LF, independent of ordering.
            self.features.append(x); self.hist_features.append(h); self.hlookup.append(lut)
        self.allowed = np.ones(len(self.y), bool)
        if a.smoke:
            self.allowed[:] = False
            # <=2000 non-GS total, stratified by split/draw/label pattern.
            groups = {}
            for r in np.flatnonzero(self.hist < 0):
                key = (int(self.split[r]), int(self.batch[r]), int(self.m[r,:12].argmax()) if self.m[r,:12].any() else 12)
                groups.setdefault(key, []).append(r)
            rng = np.random.default_rng(17054)
            for key in groups: groups[key] = rng.permutation(groups[key])
            selected = []; level = 0
            while len(selected) < 2000:
                added = False
                for key in sorted(groups):
                    if level < len(groups[key]): selected.append(int(groups[key][level])); added = True
                    if len(selected) == 2000: break
                if not added: break
                level += 1
            self.allowed[selected] = True
            for b in (0, 1):
                for s in (0, 1, 2):
                    r = np.flatnonzero((self.hist >= 0) & (self.batch == b) & (self.split == s))
                    self.allowed[rng.permutation(r)[:50]] = True
        self.train = self.rows(0)
        self.val = self.rows(1); self.test = self.rows(2)
        assert len(self.train) and len(self.val) and len(self.test)
        self.test_batch = int(a.holdout[-1]); self.train_batch = 1 - self.test_batch
        self.gs_train = np.flatnonzero(self.allowed & (self.hist >= 0) & (self.batch == self.train_batch) & (self.split == 0))
        self.gs_val = np.flatnonzero(self.allowed & (self.hist >= 0) & (self.batch == self.train_batch) & (self.split == 1))
        # Full held-out slide: all FOV splits. Never used by any training or selection.
        self.gs_test = np.flatnonzero(self.allowed & (self.hist >= 0) & (self.batch == self.test_batch))
        self.mu, self.sd = self.stats(self.train)
        self.H = None
        self.audit = dict(n_lf_train=len(self.train), n_lf_val=len(self.val), n_lf_test=len(self.test),
                          n_gs_train=len(self.gs_train), n_gs_val=len(self.gs_val), n_gs_test=len(self.gs_test),
                          qc_mask_removed=int(bad.sum()), raw_single_autofl_pairs=int((self.raw_m[:,:8].any(1) & self.raw_m[:,12]).sum()),
                          smoke=a.smoke, gs_in_lf_training=0, test_slide_in_gs_training=0)
    def rows(self, split, batch=None):
        m = self.allowed & (self.hist < 0) & (self.split == split) & self.m.any(1)
        if batch is not None: m &= self.batch == batch
        return np.flatnonzero(m)
    def raw(self, rows):
        cols = []
        for x, h, lut in zip(self.features, self.hist_features, self.hlookup):
            v = np.asarray(x[rows], np.float32).copy(); hit = self.hist[rows] >= 0
            if hit.any(): v[hit] = np.asarray(h[lut[rows[hit]]], np.float32)
            cols.append(v)
        return np.concatenate(cols, axis=1)
    def stats(self, rows):
        n = len(rows); total = np.zeros(1024, np.float64); square = total.copy()
        for i in range(0, n, 4096):
            x = self.raw(rows[i:i+4096]).astype(np.float64); total += x.sum(0); square += (x*x).sum(0)
        mu = total / n; sd = np.sqrt(np.maximum((square - n*mu*mu) / max(n-1,1), 0)) + 1e-6
        return mu.astype(np.float32), sd.astype(np.float32)
    def x(self, rows, dev, stats=None):
        mu, sd = (self.mu, self.sd) if stats is None else stats
        return torch.as_tensor((self.raw(rows) - mu) / sd, device=dev)
    def labels(self, rows, dev):
        return torch.as_tensor(self.y[rows], device=dev), torch.as_tensor(self.m[rows], device=dev)
    def images(self, rows):
        if self.H is None: self.H = np.load(Path(self.a.data) / 'pack/histology.npy', mmap_mode='r')
        return torch.from_numpy(np.asarray(self.H[self.hist[rows]], np.int16).copy())
    def gs_scale(self):
        if self.H is None: self.H = np.load(Path(self.a.data) / 'pack/histology.npy', mmap_mode='r')
        # Unlike legacy H[:500:5], fit scale only to training-slide split-0.
        r = self.gs_train[np.linspace(0, len(self.gs_train)-1, min(100,len(self.gs_train))).astype(int)]
        return float(np.percentile(self.H[self.hist[r]], 99.9))


class GS(nn.Module):
    def __init__(self, a, pretrained=True, baseline=False):
        super().__init__()
        import timm
        if a.smoke and not baseline:
            self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0,
                                         img_size=32, depth=2)
        else:
            self.vit = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=0, img_size=224)
            if pretrained:
                sd = torch.load(Path(a.data)/'models/DinoBloom-B.pth', map_location='cpu', weights_only=True)['teacher']
                self.vit.load_state_dict({k[9:]: v for k,v in sd.items() if k.startswith('backbone.') and k != 'backbone.mask_token'}, strict=True)
        self.PG = legacy_mlp(self.vit.num_features, a.dim)
        for p in self.vit.patch_embed.parameters(): p.requires_grad_(False)
        self.vit.cls_token.requires_grad_(False); self.vit.pos_embed.requires_grad_(False)
        for i, b in enumerate(self.vit.blocks):
            for p in b.parameters(): p.requires_grad_(i >= (0 if a.smoke and not baseline else 8))
    def forward(self, x):
        with torch.autocast('cuda', torch.bfloat16, enabled=x.is_cuda): h = self.vit(x).float()
        return F.normalize(self.PG(h), dim=-1)


# train_gs_bind.py pipeline copied verbatim (no optional Macenko in this experiment).
# Device / output resolution are arguments so CPU smoke uses the same augmentation.
def pipeline(x, train, smoke=False):
    dev = x.device
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    B = len(x)
    if train:
        od = -torch.log(x.clamp(min=1e-3))
        od = od * torch.exp(0.25 * torch.randn(B, 3, 1, 1, device=dev))
        od = od * torch.exp(0.15 * torch.randn(B, 1, 1, 1, device=dev))
        x = torch.exp(-od)
        x = x * (1 + 0.04 * torch.randn(B, 3, 1, 1, device=dev))
        g = torch.exp(0.2 * torch.randn(B, 1, 1, 1, device=dev)); x = x.clamp(1e-4, 1) ** g
        sig = 2.0 * torch.rand(B, device=dev); k = torch.arange(-6, 7, device=dev).float()
        ker = torch.exp(-0.5 * (k[None] / sig.clamp(min=0.05)[:, None]) ** 2); ker = ker / ker.sum(1, keepdim=True)
        kk = ker[:, None, :].expand(B, 3, 13).reshape(B * 3, 1, 1, 13)
        y = F.conv2d(F.pad(x.reshape(1, B * 3, 398, 398), (6, 6, 0, 0), mode='reflect'), kk, groups=B * 3)
        x = F.conv2d(F.pad(y, (0, 0, 6, 6), mode='reflect'), kk.transpose(2, 3), groups=B * 3).reshape(B, 3, 398, 398)
        x = x + 0.02 * torch.randn_like(x)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 398, device=dev), torch.linspace(-1, 1, 398, device=dev), indexing='ij')
        gvec = 0.08 * torch.randn(B, 2, device=dev); x = x * (1 + gvec[:, 0, None, None, None] * xx + gvec[:, 1, None, None, None] * yy)
        c = int(torch.randint(3, (1,))); sg = float(1.5 * torch.rand(1))
        if sg > 0.2:
            kk1 = torch.exp(-0.5 * (k / sg) ** 2); kk1 = (kk1 / kk1.sum()).view(1, 1, 1, 13)
            xc = x[:, c:c + 1]; xc = F.conv2d(F.pad(xc, (6, 6, 0, 0), mode='reflect'), kk1); xc = F.conv2d(F.pad(xc, (0, 0, 6, 6), mode='reflect'), kk1.transpose(2, 3))
            x = torch.cat([x[:, :c], xc, x[:, c + 1:]], 1)
        s = (0.8 + 0.2 * torch.rand(B, device=dev)); t = (torch.rand(B, 2, device=dev) - 0.5) * 0.2
        th = torch.zeros(B, 2, 3, device=dev); th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, :, 2] = t
        x = F.grid_sample(x, F.affine_grid(th, (B, 3, 224, 224), align_corners=False), padding_mode='reflection', align_corners=False)
        k90 = int(torch.randint(4, (1,))); x = torch.rot90(x, k90, (2, 3))
        if torch.rand(1) < 0.5: x = x.flip(3)
    else: x = F.interpolate(x, size=224, mode='bilinear', antialias=True)
    x = (x.clamp(0, 1) - MEAN) / STD
    return F.interpolate(x, size=32, mode='bilinear', antialias=True) if smoke else x


def prep(images, scale, dev, train, smoke):
    return pipeline((images.to(dev).float() / scale).clamp(0,1).permute(0,3,1,2), train, smoke)


def frozen(module):
    module.eval(); module.requires_grad_(False); return module


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])


def atomic_save(obj, path):
    path = Path(path); tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); os.replace(tmp, path)


class Trainer:
    """Single ckpt.pt holds all stage/job cursors, optimizer/scheduler and RNG states.
    A completed stage exports its own immutable artifact; resume never restarts it.
    """
    def __init__(self, a, key, net, lr=None):
        self.a, self.key, self.net = a, key, net
        self.path = Path(a.out)/'ckpt.pt'; self.bundle = torch.load(self.path, map_location='cpu', weights_only=False) if self.path.exists() else {'jobs':{}, 'config':vars(a).copy()}
        contract = ('arm','holdout','seed','smoke','dim','steps','bs','lr','sigma_bridge','w_sig','targets','lf_marker','lf_autofl','data','lf_lr','flow_lr','tau','lf_bs','eval_every','threads')
        for k in contract:
            if self.bundle['config'][k] != vars(a)[k]: raise ValueError(f'checkpoint configuration changed: {k}')
        self.opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr or a.lr, weight_decay=.05)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, lr or a.lr, total_steps=a.steps, pct_start=.1 if a.smoke else .05)
        self.start = 0; self.best = float('inf'); self.best_state = None; self.log = []
        if key in self.bundle['jobs']:
            ck = self.bundle['jobs'][key]; net.load_state_dict(ck['net']); self.opt.load_state_dict(ck['opt'])
            # Optimizer was loaded on CPU to keep RNG tensors CPU; move momentum.
            dev = next(net.parameters()).device
            for state in self.opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v): state[k] = v.to(dev)
            self.sched.load_state_dict(ck['sched']); self.start=ck['step']; self.best=ck['best']; self.best_state=ck['best_state']; self.log=ck['log']; restore_rng(ck['rng'])
            print(f'resume {key} at {self.start}/{a.steps}', flush=True)
    def save(self, step):
        self.bundle['jobs'][self.key] = dict(net=self.net.state_dict(), opt=self.opt.state_dict(), sched=self.sched.state_dict(),
                      step=step, best=self.best, best_state=self.best_state, log=self.log, rng=rng_state())
        atomic_save(self.bundle, self.path)
    def run(self, loss_fn, validate):
        for step in range(self.start, self.a.steps):
            self.net.train(); loss = loss_fn(); self.opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1); self.opt.step(); self.sched.step()
            interval = 10 if self.a.smoke else self.a.eval_every
            if (step+1) % interval == 0 or step+1 == self.a.steps:
                self.net.eval()
                with torch.no_grad(): val = float(validate())
                if np.isfinite(val) and val < self.best:
                    self.best=val; self.best_state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
                rec=dict(job=self.key, step=step+1, loss=float(loss.detach()), val_loss=val)
                self.log.append(rec); print(json.dumps(rec), flush=True)
                self.save(step+1)
            if STOP:
                self.save(step+1); raise SystemExit(99)
        if self.best_state is None: raise RuntimeError('no finite validation score')
        self.net.load_state_dict(self.best_state); self.net.eval()
        return self.log


def sample(rows, n): return np.random.choice(rows, min(n,len(rows)), replace=False)


def export(a, name, net, extra=None):
    atomic_save(dict(state=net.state_dict(), args=vars(a), **(extra or {})), Path(a.out)/name)


def load_spaces(a, dev):
    ck = torch.load(Path(a.out)/'lf_spaces.pt', map_location='cpu', weights_only=False)
    assert ck['args']['arm'] == a.arm and ck['args']['smoke'] == a.smoke
    m = Spaces(a.dim).to(dev); m.load_state_dict(ck['state']); return frozen(m)


def train_lf(a, d, dev):
    net = Spaces(a.dim).to(dev); t = Trainer(a,'lf_spaces',net, a.lf_lr)
    def loss(rows, training):
        x=d.x(rows,dev); y,m=d.labels(rows,dev); z=net.lf(x); zm=net.marker(y,m)
        base=mse_mask(net.decode(z),y,m) + net.sig(z,zm,near_duplicate(y,m,a.tau))
        if a.arm=='A3': return base
        # Each decoded entry is removed from its marker input, including autofl.
        # Expand 13 leave-one-out views, then score only the removed entries.
        yy=y[:,None,:].expand(-1,13,-1).reshape(-1,13)
        mm=m[:,None,:].expand(-1,13,-1).clone()
        mm[:,torch.arange(13),torch.arange(13)]=False
        pred=net.decode(net.marker(yy,mm.reshape(-1,13))).reshape(-1,13,13).diagonal(dim1=1,dim2=2)
        return base + mse_mask(pred,y,m)
    val=np.random.default_rng(54).permutation(d.val)[:min(len(d.val),2048)]
    logs=t.run(lambda:loss(sample(d.train,a.lf_bs),True),lambda:loss(val,False))
    export(a,'lf_spaces.pt',net,dict(mu=d.mu,sd=d.sd,audit=d.audit,log=logs))
    fit_reference(a,d,dev)


def ridge_fit(d, rows, dev, mu, sd, dims=None, y=None, m=None, max_rows=60000):
    y = d.y if y is None else y; m=d.m if m is None else m
    p=1024 if dims is None else dims[1]-dims[0]
    wt=np.zeros((p+1,13),np.float32)
    for j in range(13):
        r=rows[m[rows,j]]; r=np.random.default_rng(j).permutation(r)[:max_rows]
        if not len(r): continue
        aa=torch.zeros((p+1,p+1),device=dev,dtype=torch.float64); ab=torch.zeros(p+1,device=dev,dtype=torch.float64)
        for i in range(0,len(r),4096):
            rr=r[i:i+4096]; x=(d.raw(rr)-mu)/sd
            if dims is not None: x=x[:,dims[0]:dims[1]]
            x=torch.as_tensor(x,device=dev,dtype=torch.float64); x=torch.cat([x,x.new_ones(len(x),1)],1)
            aa+=x.T@x; ab+=x.T@torch.as_tensor(y[rr,j],device=dev,dtype=torch.float64)
        reg=torch.eye(p+1,device=dev,dtype=torch.float64)*10; reg[-1,-1]=0
        wt[:,j]=torch.linalg.solve(aa+reg,ab).float().cpu().numpy()
    return wt


def reference_signature(a,d):
    h=hashlib.sha256(); h.update(d.train.tobytes()); h.update(d.mu.tobytes()); h.update(d.sd.tobytes())
    h.update(d.y[d.train].tobytes()); h.update(d.m[d.train].tobytes())
    for f in d.features:
        st=Path(f.filename).stat(); h.update(str((str(f.filename),st.st_size,st.st_mtime_ns)).encode())
    for s in (a.targets,a.lf_marker,a.lf_autofl,'K019-batch1-group8-masked-ridge10-v1'): h.update(s.encode())
    return h.hexdigest()


def fit_reference(a,d,dev):
    path=Path(a.ridge_cache or (Path(a.out)/'ridge.npz')); signature=reference_signature(a,d)
    if path.exists():
        with np.load(path) as r: assert str(r['signature'])==signature,'ridge cache mismatch'
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    wt=ridge_fit(d,d.train,dev,d.mu,d.sd)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp.npz'); np.savez(tmp,weight=wt,mu=d.mu,sd=d.sd,signature=signature,train_rows=d.train,names=d.names)
    os.replace(tmp,path); print(f'fixed ridge saved: {path}',flush=True)


def ridge_predict(d, rows, ref):
    x=(d.raw(rows)-ref['mu'])/ref['sd']; return x@ref['weight'][:-1]+ref['weight'][-1]


def train_gs(a,d,dev):
    if a.arm=='A0': raise ValueError('A0 is eval-only')
    spaces=load_spaces(a,dev); gs=GS(a).to(dev)
    if a.arm=='A3': net=nn.ModuleDict({'gs':gs,'sig':SigLIP()}).to(dev)
    else: net=nn.ModuleDict({'gs':gs,'P2':mlp(1024,a.dim),'sig':SigLIP()}).to(dev)
    scale=d.gs_scale(); t=Trainer(a,'gs_encoder',net)
    def loss(rows,training):
        zg=net['gs'](prep(d.images(rows),scale,dev,training,a.smoke))
        with torch.no_grad(): x=d.x(rows,dev)
        if a.arm=='A3':
            with torch.no_grad(): zl=spaces.lf(x)
            return F.mse_loss(zg,zl)+a.w_sig*net['sig'](zg,zl)
        zl=F.normalize(net['P2'](x),dim=-1)
        return net['sig'](zg,zl)
    def val():
        total=0.; n=0
        for i in range(0,len(d.gs_val),a.bs):
            r=d.gs_val[i:i+a.bs]; total+=float(loss(r,False))*len(r); n+=len(r)
        return total/max(n,1)
    logs=t.run(lambda:loss(sample(d.gs_train,a.bs),True),val)
    export(a,'gs_encoder.pt',net,dict(scale=scale,mu=d.mu,sd=d.sd,log=logs,audit=d.audit))


def load_gs(a,dev):
    ck=torch.load(Path(a.out)/'gs_encoder.pt',map_location='cpu',weights_only=False)
    if a.arm=='A3': net=nn.ModuleDict({'gs':GS(a,False),'sig':SigLIP()}).to(dev)
    else: net=nn.ModuleDict({'gs':GS(a,False),'P2':mlp(1024,a.dim),'sig':SigLIP()}).to(dev)
    net.load_state_dict(ck['state']); return frozen(net), ck


def train_flow(a,d,dev):
    if a.arm not in ('A1','A2'):
        json_write(Path(a.out)/'flow.json',dict(status='not_applicable',arm=a.arm)); return
    spaces=load_spaces(a,dev); gs,_=load_gs(a,dev); net=Velocity(a.dim).to(dev); t=Trainer(a,'flow',net,a.flow_lr)
    def endpoints(rows, target_rows=None):
        with torch.no_grad():
            z2=F.normalize(gs['P2'](d.x(rows,dev)),dim=-1)
            if a.arm=='A1': z1=spaces.lf(d.x(rows,dev))
            else:
                y,m=d.labels(target_rows,dev); z1=spaces.marker(y,m)
                from scipy.optimize import linear_sum_assignment
                cost=torch.cdist(z2,z1).square().cpu().numpy()
                # Exclude even chance same-cell correspondences before OT.
                cost[np.asarray(rows)[:,None]==np.asarray(target_rows)[None,:]]=1e6
                ii,jj=linear_sum_assignment(cost); assert np.array_equal(ii,np.arange(len(rows)))
                assert np.all(np.asarray(rows)!=np.asarray(target_rows)[jj])
                z1=z1[torch.as_tensor(jj,device=dev)]
        return z2,z1
    def objective(rows,target_rows=None,validation=False):
        z2,z1=endpoints(rows,target_rows)
        # Stochastic interpolant: x_t=(1-t)x0+t*x1+sigma*t*(1-t)*eps.
        # Analytic path derivative has bounded endpoint variance.
        if validation:
            gen=torch.Generator(device=dev).manual_seed(54)
            tt=torch.rand((len(rows),1),generator=gen,device=dev); eps=torch.randn(z1.shape,generator=gen,device=dev)
        else: tt=torch.rand(len(rows),1,device=dev); eps=torch.randn_like(z1)
        xt=(1-tt)*z2+tt*z1+a.sigma_bridge*tt*(1-tt)*eps
        velocity=z1-z2+a.sigma_bridge*(1-2*tt)*eps
        return F.mse_loss(net(xt,tt),velocity)
    n=min(a.lf_bs,len(d.val)); vr=np.random.default_rng(54).permutation(d.val)[:n]; vt=np.roll(vr,1)
    # Marginal target population contains observed marker rows, not GS or empty rows.
    logs=t.run(lambda:objective(sample(d.train,a.lf_bs),sample(d.train,a.lf_bs)),lambda:objective(vr,vt,True))
    export(a,'flow.pt',net,dict(log=logs))

class Predictor:
    """Common inference interface used by internal and clinical evaluation."""
    def __init__(self,a,d,dev):
        self.a,self.d,self.dev=a,d,dev
        if a.arm=='A0':
            self.base=load_baseline(a,dev)
            # Baseline evaluation preprocessing is exactly the original TT6c.
            self.scale=float(np.percentile(np.load(Path(a.data)/'pack/histology.npy',mmap_mode='r')[:500:5],99.9))
            self.legacy_ref=legacy_teacher(a,d,dev)
        else:
            self.spaces=load_spaces(a,dev); self.gs,ck=load_gs(a,dev); self.scale=ck['scale']
            if a.arm in ('A1','A2'):
                self.flow=Velocity(a.dim).to(dev)
                self.flow.load_state_dict(torch.load(Path(a.out)/'flow.pt',map_location='cpu',weights_only=False)['state']); frozen(self.flow)
    @torch.no_grad()
    def images(self,x):
        z=self.base['gs'](x) if self.a.arm=='A0' else self.gs['gs'](x)
        if self.a.arm=='A0': p=self.base['S'](z*math.sqrt(self.a.dim)); mapped=z
        else:
            mapped=integrate(self.flow,z,self.a.euler_steps) if self.a.arm in ('A1','A2') else z
            p=self.spaces.decode(mapped)
        return p.cpu().numpy(),z.cpu().numpy(),mapped.cpu().numpy()
    @torch.no_grad()
    def lf(self,rows):
        if self.a.arm=='A0':
            x=(self.d.raw(rows)-self.legacy_ref['mu'])/self.legacy_ref['sd']
            zl=F.normalize(self.base['PL'](torch.as_tensor(x[:,:512],device=self.dev)),dim=-1)
            return ridge_predict(self.d,rows,self.legacy_ref),zl.cpu().numpy()
        x=self.d.x(rows,self.dev); z1=self.spaces.lf(x); p=self.spaces.decode(z1)
        match=z1 if self.a.arm=='A3' else F.normalize(self.gs['P2'](x),dim=-1)
        return p.cpu().numpy(),match.cpu().numpy()


def load_baseline(a,dev):
    path=Path(a.baseline or (Path(a.data)/f'runs/n029_tt6c_x_{a.holdout}/gs_bind.pt'))
    state=torch.load(path,map_location='cpu',weights_only=True)
    gs=GS(a,False,True); gs.load_state_dict({k:v for k,v in state.items() if k.startswith(('vit.','PG.'))})
    net=nn.ModuleDict({'gs':gs,'PL':legacy_mlp(512,a.dim),'S':legacy_mlp(a.dim,13)}).to(dev)
    for k in ('PL','S'): net[k].load_state_dict({key[len(k)+1:]:v for key,v in state.items() if key.startswith(k+'.')})
    return frozen(net)


def legacy_teacher(a,d,dev):
    path=Path(a.out)/'baseline_teacher.npz'
    if path.exists(): return dict(np.load(path))
    # Recreate original TT6c teachers (including their original masks); fixed
    # reference ridge and all new arms use the documented K019 mask instead.
    w=Path(a.data); t=np.load(w/'targets_v4c.npz'); aux=np.load(w/'targets_v6c.npz')
    for k in ('hist_row','split','batch','slide'): assert np.array_equal(t[k],aux[k])
    rows=np.flatnonzero(d.allowed & (d.hist<0) & (d.split==0))
    normrows=rows[t['mask'][rows,:12].any(1)]; mu,sd=d.stats(normrows)
    primary=ridge_fit(d,rows,dev,mu,sd,dims=(0,512),y=t['Y'],m=t['mask'])
    secondary=ridge_fit(d,rows,dev,mu,sd,dims=(512,1024),y=aux['Y'],m=aux['mask'])
    wt=np.zeros((1025,13),np.float32); wt[:512,:12]=primary[:-1,:12]; wt[-1,:12]=primary[-1,:12]
    wt[512:1024,12]=secondary[:-1,12]; wt[-1,12]=secondary[-1,12]
    np.savez(path,weight=wt,mu=mu,sd=sd,train_rows=rows); return dict(np.load(path))


def geometry(a,b,k=10):
    """Linear CKA and overlap of *mutual* cosine-kNN graphs on corresponding rows.
    Also report directed overlap, since mutual and directed kNN differ.
    Chunked similarities keep peak memory O(512*N), not O(N^2).
    """
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64); n=len(a)
    if n<3: return dict(n=n,linear_cka=None,mutual_knn_overlap=None,directed_knn_overlap=None)
    ac=a-a.mean(0); bc=b-b.mean(0)
    cka=float(np.square(ac.T@bc).sum()/np.sqrt(np.square(ac.T@ac).sum()*np.square(bc.T@bc).sum()).clip(min=1e-20))
    k=min(k,n-1)
    def graph(x):
        x=x/np.linalg.norm(x,axis=1,keepdims=True).clip(min=1e-12); edges=set()
        for s in range(0,n,512):
            sim=x[s:s+512]@x.T; sim[np.arange(len(sim)),np.arange(s,s+len(sim))]=-np.inf
            nb=np.argpartition(-sim,k-1,axis=1)[:,:k]
            for i,nn_ in enumerate(nb): edges.update((s+i,int(j)) for j in nn_)
        return edges
    ea,eb=graph(a),graph(b)
    ma={(i,j) for i,j in ea if (j,i) in ea}; mb={(i,j) for i,j in eb if (j,i) in eb}
    # overlap coefficient = common mutual edges / mean edge count.
    return dict(n=n,k=k,linear_cka=cka,directed_knn_overlap=len(ea&eb)/(n*k),
                mutual_knn_overlap=2*len(ma&mb)/max(len(ma)+len(mb),1),mutual_edges_a=len(ma)//2,mutual_edges_b=len(mb)//2)


def retrieval(a,b):
    a=np.asarray(a,np.float32); b=np.asarray(b,np.float32)
    a=a/np.linalg.norm(a,axis=1,keepdims=True).clip(min=1e-12)
    b=b/np.linalg.norm(b,axis=1,keepdims=True).clip(min=1e-12)
    ranks=[]
    for s in range(0,len(a),512):
        sim=a[s:s+512]@b.T; diag=sim[np.arange(len(sim)),np.arange(s,s+len(sim))]
        # Ties count pessimistically, matching train_gs_bind.py.
        ranks.extend((sim>=diag[:,None]).sum(1)-1)
    ranks=np.asarray(ranks)
    return dict(n=len(a),R_at_1=float((ranks<1).mean()),R_at_10=float((ranks<10).mean()),
                median_rank=float(np.median(ranks+1)),chance_R_at_1=1/len(a),chance_R_at_10=min(10/len(a),1))


def run_eval(a,d,dev):
    fit_reference(a,d,dev); ref=dict(np.load(a.ridge_cache or (Path(a.out)/'ridge.npz')))
    p=Predictor(a,d,dev); rows=d.gs_val if a.validation_only else d.gs_test
    output='validation' if a.validation_only else 'eval'
    metrics=[]
    path=Path(a.out)/'ckpt.pt'
    bundle=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {}
    fingerprints=[]
    files=([Path(a.baseline or Path(a.data)/f'runs/n029_tt6c_x_{a.holdout}/gs_bind.pt')] if a.arm=='A0' else
           [Path(a.out)/f for f in ('lf_spaces.pt','gs_encoder.pt','flow.pt')])
    for file in files:
        if file.exists(): fingerprints.append((str(file),file.stat().st_size,file.stat().st_mtime_ns))
    signature=(fingerprints,rows.tolist(),a.smoke,a.euler_steps)
    progress=bundle.setdefault('evaluations',{}).get(output)
    if progress is None or progress['signature']!=signature:
        progress=dict(signature=signature,done=0,parts=[[] for _ in range(7)])
    gs_pred,zgs,zmapped,lf_pred,zlf,fixed,zlf1=progress['parts']
    with torch.no_grad():
        for s in range(progress['done'],len(rows),a.bs):
            rr=rows[s:s+a.bs]
            x=prep(d.images(rr),p.scale,dev,False,a.smoke and a.arm!='A0')
            pp,zz,mm=p.images(x); own,zl=p.lf(rr)
            gs_pred.append(pp); zgs.append(zz); zmapped.append(mm); lf_pred.append(own); zlf.append(zl); fixed.append(ridge_predict(d,rr,ref))
            if a.arm in ('A1','A2'): zlf1.append(p.spaces.lf(d.x(rr,dev)).cpu().numpy())
            progress['done']=s+len(rr)
            if STOP or progress['done']==len(rows) or (s//a.bs+1)%20==0:
                bundle['evaluations'][output]=progress; atomic_save(bundle,path)
            if STOP: raise SystemExit(99)
    pred,zg,mapped,own,zl,teacher=[np.concatenate(v) for v in (gs_pred,zgs,zmapped,lf_pred,zlf,fixed)]
    j=12; observed=d.m[rows,j]
    result=dict(arm=a.arm,holdout=a.holdout,smoke=a.smoke,validation_only=a.validation_only,
                M1_autofl_truth_r=corr(pred[observed,j],d.y[rows[observed],j]),
                M3=retrieval(zg,zl),M4_GS_LF=geometry(zg,zl),audit=d.audit)
    if a.arm in ('A1','A2'):
        result['M4_mapped_GS_LFspace1']=geometry(mapped,np.concatenate(zlf1))
    metrics.append(dict(metric='M1',target='autofl',path='GS',r=result['M1_autofl_truth_r'],n=int(observed.sum())))
    tb=d.train_batch if a.validation_only else d.test_batch
    ev=d.val if a.validation_only else d.test
    ev=ev[d.batch[ev]==tb]
    # LF test predictions are saved in full; same rows and ridge across all arms.
    ep=[]; rp=[]
    for s in range(0,len(ev),a.lf_bs):
        rr=ev[s:s+a.lf_bs]; ep.append(p.lf(rr)[0]); rp.append(ridge_predict(d,rr,ref))
    ep=np.concatenate(ep); rp=np.concatenate(rp)
    for k,name in enumerate(d.names):
        obs=d.m[ev,k]
        for label,tch,tch_ev in [('fixed_ridge',teacher,rp),('own_LF',own,ep)]:
            rst=corr(pred[:,k],tch[:,k]); rtt=corr(tch_ev[obs,k],d.y[ev[obs],k])
            metrics.append(dict(metric='M2',target=name,path=label,r_student_teacher=rst,r_teacher_truth=rtt,
                                r_est=product(rst,rtt),n_gs=len(rows),n_truth=int(obs.sum()),estimate_assumption='CIA; not marker truth'))
        if k==12:
            rtt=corr(teacher[observed,k],d.y[rows[observed],k]); rst=corr(pred[:,k],teacher[:,k])
            metrics.append(dict(metric='M2_autofl_GS_check',target=name,path='fixed_ridge',r_est=product(rst,rtt),
                                r_teacher_truth=rtt,r_student_teacher=rst,r_measured=result['M1_autofl_truth_r']))
        for label,pp in [('own_LF',ep),('fixed_ridge',rp)]:
            metrics.append(dict(metric='M5_same_draw',target=name,path=label,r=corr(pp[obs,k],d.y[ev[obs],k]),n=int(obs.sum())))
    if not a.validation_only:
        # M5 includes BOTH draws' split-2 cells; additional per-draw rows preserve shift information.
        alle=d.test; allp=[]; allr=[]
        for s in range(0,len(alle),a.lf_bs):
            rr=alle[s:s+a.lf_bs]; allp.append(p.lf(rr)[0]); allr.append(ridge_predict(d,rr,ref))
        allp=np.concatenate(allp); allr=np.concatenate(allr)
        for b in ('both',0,1):
            for k,name in enumerate(d.names):
                obs=d.m[alle,k] & (True if b=='both' else d.batch[alle]==b)
                for label,pp in [('own_LF',allp),('fixed_ridge',allr)]:
                    metrics.append(dict(metric='M5',target=name,path=label,batch=b,r=corr(pp[obs,k],d.y[alle[obs],k]),n=int(obs.sum())))
        np.savez(Path(a.out)/'lf_test_preds.npz',rows=alle,pred=allp,fixed_ridge=allr,truth=d.y[alle],mask=d.m[alle],batch=d.batch[alle],names=d.names)
        if a.arm in ('A1','A2'):
            z1=[]; z2=[]
            for s in range(0,len(alle),a.lf_bs):
                x=d.x(alle[s:s+a.lf_bs],dev)
                with torch.no_grad(): z1.append(p.spaces.lf(x).cpu().numpy()); z2.append(F.normalize(p.gs['P2'](x),dim=-1).cpu().numpy())
            z1=np.concatenate(z1); z2=np.concatenate(z2)
            result['M4_LF_space1_space2']=geometry(z1,z2)
            np.savez(Path(a.out)/'lf_geometry.npz',rows=alle,z1=z1,z2=z2)
    np.savez(Path(a.out)/f'{output}_gs_preds.npz',rows=rows,hist_row=d.hist[rows],pred=pred,fixed_ridge=teacher,own_lf=own,
             z_gs=zg,z_matching_lf=zl,z_mapped=mapped,truth=d.y[rows],mask=d.m[rows],names=d.names)
    result['per_target']=metrics
    json_write(Path(a.out)/f'{output}.json',result)
    flat=metrics.copy()
    for group in ('M3','M4_GS_LF','M4_LF_space1_space2','M4_mapped_GS_LFspace1'):
        if group in result:
            flat += [dict(metric=group,statistic=k,value=v) for k,v in result[group].items()]
    csv_write(Path(a.out)/f'{output}.csv',flat)
    print(json.dumps(dict(output=output,arm=a.arm,M1=result['M1_autofl_truth_r'],M3=result['M3'])),flush=True)


class Completion(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.spaces=Spaces(dim); self.fused=mlp(2*dim,13)
    def forward(self,x,y,m):
        z=self.spaces.lf(x); em=self.spaces.marker(y,m)
        return self.spaces.decode(z),self.fused(torch.cat([z,em],-1)*math.sqrt(self.spaces.dim))


def completion_stage(a,d,dev):
    dest=Path(a.out)/'completion'; dest.mkdir(exist_ok=True)
    report=[]; coherence=[]
    # Every model (including normalization and EM) is refit inside the protocol.
    for protocol,trainb,testb in [('fov',None,None),('draw0_to_1',0,1),('draw1_to_0',1,0)]:
        train=d.rows(0,trainb); val=d.rows(1,trainb); test=d.rows(2,testb)
        assert np.all(d.hist[train]<0) and np.all(d.split[train]==0)
        if trainb is not None: assert np.all(d.batch[train]==trainb) and not np.intersect1d(train,test).size
        stats=d.stats(train); seed_all(a.seed+{'fov':100,'draw0_to_1':200,'draw1_to_0':300}[protocol])
        net=Completion(a.dim).to(dev); trainer=Trainer(a,'completion_'+protocol,net,a.lf_lr)
        def objective(rows,validation=False):
            x=d.x(rows,dev,stats); y,m=d.labels(rows,dev)
            if validation:
                gen=torch.Generator(device=dev).manual_seed(54); keep=(torch.rand(m.shape,device=dev,generator=gen)>.5)&m
            else: keep=(torch.rand(m.shape,device=dev)>.5)&m
            pl,pf=net(x,y,keep)
            z=net.spaces.lf(x); em=net.spaces.marker(y,m)
            return mse_mask(pl,y,m)+mse_mask(pf,y,m)+net.spaces.sig(z,em,near_duplicate(y,m,a.tau))
        vr=np.random.default_rng(54).permutation(val)[:min(1024,len(val))]
        logs=trainer.run(lambda:objective(sample(train,a.lf_bs)),lambda:objective(vr,True))
        atomic_save(dict(state=net.state_dict(),mu=stats[0],sd=stats[1],train_rows=train,val_rows=val,test_rows=test,
                         protocol=protocol,args=vars(a),log=logs),dest/f'{protocol}.pt')
        single=d.m[test,:8].sum(1)==1; pooled=d.m[test,8:12].any(1)
        c1=test[single & d.m[test,12]]
        if len(c1)==0:
            report.append(dict(test='C1',protocol=protocol,target='autofl',n=0,status='unavailable: no single-marker cells with observed autofl in targets_tt6c'))
        tasks=[('C1',12,c1)] + [('C2',j,test[pooled & d.m[test,j]]) for j in range(8,12)]
        for typ,j,rows in tasks:
            if len(rows)==0: continue
            p0=[]; p1=[]; inputs=[]
            for s in range(0,len(rows),a.lf_bs):
                rr=rows[s:s+a.lf_bs]; y,m=d.labels(rr,dev); mi=m.clone()
                # C1: only the observed single surface marker; C2: only OTHER groups.
                if typ=='C1': mi[:,8:]=False
                else: mi[:,:8]=False; mi[:,12]=False
                mi[:,j]=False; assert not mi[:,j].any()
                with torch.no_grad(): l,f=net(d.x(rr,dev,stats),y,mi)
                p0.append(l.cpu().numpy()); p1.append(f.cpu().numpy()); inputs.append(mi.cpu().numpy())
            p0=np.concatenate(p0); p1=np.concatenate(p1); ins=np.concatenate(inputs)
            # Report draw-2 failed group's absence, rather than manufacturing labels.
            nt=int(d.m[train,j].sum())
            r0=corr(p0[:,j],d.y[rows,j]) if nt else None; r1=corr(p1[:,j],d.y[rows,j]) if nt else None
            report.append(dict(test=typ,protocol=protocol,target=d.names[j],n=len(rows),r_lf=r0,r_fused=r1,
                               delta=None if r0 is None or r1 is None else r1-r0,n_train_target=nt,status='ok' if nt else 'unavailable: no observed source-draw training labels',mean_observed_inputs=float(ins.sum(1).mean())))
            np.savez(dest/f'{protocol}_{typ}_{j}.npz',rows=rows,lf=p0,fused=p1,truth=d.y[rows],mask=d.m[rows],input_mask=ins,names=d.names)
        # C3 descriptive only: single-marker evaluation population; no truth inference.
        pop=test[single]; lfs=[]; fus=[]
        for s in range(0,len(pop),a.lf_bs):
            rr=pop[s:s+a.lf_bs]; y,m=d.labels(rr,dev); m=m.clone(); m[:,8:]=False
            with torch.no_grad(): l,f=net(d.x(rr,dev,stats),y,m)
            lfs.append(l.cpu().numpy()); fus.append(f.cpu().numpy())
        pl=np.concatenate(lfs); pf=np.concatenate(fus)
        np.savez(dest/f'{protocol}_C3.npz',rows=pop,lf=pl,fused=pf,observed_values=d.y[pop],mask=d.m[pop],names=d.names)
        for marker in ('single:CD19','single:CD3','single:CD14','single:CD56'):
            j=d.names.index(marker); observed=d.m[pop,j]
            if not observed.any(): continue
            cutoff=float(np.quantile(d.y[pop[observed],j],.95)); hi=observed&(d.y[pop,j]>=cutoff)
            for k,name in enumerate(d.names):
                if k==j: continue
                coherence.append(dict(protocol=protocol,conditioning_marker=marker,decoded_target=name,n_high=int(hi.sum()),
                    observed_threshold=cutoff,lf_high_mean=float(pl[hi,k].mean()),fused_high_mean=float(pf[hi,k].mean()),
                    lf_population_mean=float(pl[:,k].mean()),fused_population_mean=float(pf[:,k].mean()),
                    lf_delta_population=float(pl[hi,k].mean()-pl[:,k].mean()),fused_delta_population=float(pf[hi,k].mean()-pf[:,k].mean()),
                    status='descriptive; other-marker truth absent'))
        print(f'completion {protocol}: trained={len(train)}, tested={len(test)}',flush=True)
    # Explicit n=0 rows for missing C2 group/draw combinations are part of the result.
    for protocol in ('fov','draw0_to_1','draw1_to_0'):
        for j in range(8,12):
            if not any(r['test']=='C2' and r['protocol']==protocol and r['target']==d.names[j] for r in report):
                report.append(dict(test='C2',protocol=protocol,target=d.names[j],n=0,status='no observed test truth after QC'))
    csv_write(dest/'completion.csv',report); csv_write(dest/'coherence.csv',coherence)
    json_write(dest/'completion.json',dict(metrics=report,coherence=coherence,audit=d.audit))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['lf_spaces','gs_encoder','flow','eval','completion'])
    p.add_argument('--out',required=True); p.add_argument('--arm',choices=['A0','A1','A2','A3'],default='A3')
    p.add_argument('--holdout',choices=['batch0','batch1'],default='batch1'); p.add_argument('--data',default=ROOT)
    p.add_argument('--targets',default='targets_tt6c.npz')
    p.add_argument('--lf-marker',default='runs/n029_r18_v4c_fov/probe'); p.add_argument('--lf-autofl',default='runs/n029_r18_v6c_fov/probe')
    p.add_argument('--steps',type=int,default=2500); p.add_argument('--bs',type=int,default=128)
    p.add_argument('--lf-bs',type=int,default=256); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lf-lr',type=float,default=1e-3); p.add_argument('--flow-lr',type=float,default=1e-3)
    p.add_argument('--dim',type=int,default=256); p.add_argument('--sigma-bridge',type=float,choices=[0,.1],default=0)
    p.add_argument('--w-sig',type=float,choices=[0,.5],default=0); p.add_argument('--euler-steps',type=int,default=32)
    p.add_argument('--tau',type=float,default=.25); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']); p.add_argument('--threads',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=250); p.add_argument('--smoke',action='store_true')
    p.add_argument('--ridge-cache',default=''); p.add_argument('--baseline',default=''); p.add_argument('--validation-only',action='store_true')
    return p


def main():
    a=parser().parse_args()
    if a.arm=='A0' and a.stage!='eval': raise ValueError('A0 is evaluation only: never retrain the baseline')
    Path(a.out).mkdir(parents=True,exist_ok=True)
    if a.smoke:
        a.steps=min(a.steps,20); a.bs=min(a.bs,4); a.lf_bs=min(a.lf_bs,16)
    torch.set_num_threads(a.threads); seed_all(a.seed)
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGUSR1,stop_handler)
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    started=time.time(); d=Data(a)
    json_write(Path(a.out)/'data_audit.json',d.audit)
    # Every stage reconstructs train-only LF normalization; check it matches artifacts.
    if a.stage!='lf_spaces' and a.arm!='A0' and (Path(a.out)/'lf_spaces.pt').exists():
        ck=torch.load(Path(a.out)/'lf_spaces.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(ck['mu'],d.mu) and np.array_equal(ck['sd'],d.sd),'LF normalization drift'
    print(json.dumps(dict(stage=a.stage,arm=a.arm,holdout=a.holdout,device=str(dev),audit=d.audit)),flush=True)
    funcs=dict(lf_spaces=train_lf,gs_encoder=train_gs,flow=train_flow,eval=run_eval,completion=completion_stage)
    funcs[a.stage](a,d,dev)
    print(json.dumps(dict(stage=a.stage,status='done',seconds=round(time.time()-started,2))),flush=True)


if __name__=='__main__': main()
