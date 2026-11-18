#!/usr/bin/env -S python3 -I -B
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
CODE_SIGNATURE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
    # Surface-labelled rows: identical observed surface sets and nearby values.
    # Autofl-only rows also contain a real scalar readout and need the same rule.
    surf, sm = y[:, :12], m[:, :12]
    same = (sm[:, None] == sm[None]).all(-1) & (sm.sum(1)[:, None] > 0)
    diff = ((surf[:, None] - surf[None]).abs() * sm[:, None]).amax(-1)
    af = m[:, 12] & ~sm.any(-1)
    ignore = (same & (diff < tau)) | (af[:,None] & af[None] & ((y[:,12,None]-y[None,:,12]).abs() < tau))
    ignore.fill_diagonal_(False)
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
                          lf_heldout_non_gs_unstained_by_holdout={f'batch{b}':dict(slides=np.unique(self.slide[(self.hist>=0)&(self.batch==b)]).tolist(),
                              split0_rows=int(np.isin(self.train,np.flatnonzero((self.hist<0)&np.isin(self.slide,np.unique(self.slide[(self.hist>=0)&(self.batch==b)])))).sum()),
                              full_split0_rows=int(((self.hist<0)&(self.split==0)&self.m.any(1)&np.isin(self.slide,np.unique(self.slide[(self.hist>=0)&(self.batch==b)]))).sum())) for b in (0,1)},
                          n_gs_train=len(self.gs_train), n_gs_val=len(self.gs_val), n_gs_test=len(self.gs_test),
                          qc_mask_removed=int(bad.sum()), raw_single_autofl_pairs=int((self.raw_m[:,:8].any(1) & self.raw_m[:,12]).sum()),
                          smoke=a.smoke, targets=a.targets,
                          gs_in_lf_training=int((self.hist[self.train]>=0).sum()),
                          test_slide_in_gs_training=int((self.batch[self.gs_train]==self.test_batch).sum()),
                          gs_train_val_overlap=int(np.intersect1d(self.gs_train,self.gs_val).size),
                          gs_train_test_overlap=int(np.intersect1d(self.gs_train,self.gs_test).size),
                          gs_train_splits=np.unique(self.split[self.gs_train]).tolist(),
                          gs_val_splits=np.unique(self.split[self.gs_val]).tolist(),
                          raw_failed_group_draw2=int((self.raw_m[:,8] & (self.batch==1)).sum()),
                          single_marker_cells=int((self.raw_m[:,:8].sum(1)==1).sum()))
        assert not self.audit['gs_in_lf_training'] and not self.audit['test_slide_in_gs_training']
        assert not self.audit['gs_train_test_overlap'] and not self.audit['gs_train_val_overlap']
        if a.targets == 'targets_v5cx_v10.npz':
            single=self.raw_m[:,:8].sum(1)==1
            assert self.raw_m[single,12].all(), 'completion single-marker autofl missing'
            assert not self.audit['raw_failed_group_draw2'], 'v10 QC entries should already be masked'
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
            check_stop()
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
        if self.bundle.get('code_signature', CODE_SIGNATURE) != CODE_SIGNATURE: raise ValueError('checkpoint code signature changed')
        self.bundle['code_signature'] = CODE_SIGNATURE
        contract = ('space_dir','gs_dir','init_flow','pair_cache','arm','holdout','seed','smoke','dim','steps','bs','lr','sigma_bridge','w_sig','targets','lf_marker','lf_autofl','data','lf_lr','flow_lr','tau','lf_bs','eval_every','threads','map_source','a2_cost','gs_map_steps')
        for k in contract:
            if self.bundle['config'][k] != vars(a)[k]: raise ValueError(f'checkpoint configuration changed: {k}')
        self.opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr or a.lr, weight_decay=.05)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.opt, lr or a.lr, total_steps=a.steps, pct_start=.1 if a.smoke else .05)
        self.start = 0; self.best = float('inf'); self.best_state = None; self.log = []
        self.pending_validation=None; self.pending_loss=None; self.validation_details={}
        if key in self.bundle['jobs']:
            ck = self.bundle['jobs'][key]; net.load_state_dict(ck['net']); self.opt.load_state_dict(ck['opt'])
            # Optimizer was loaded on CPU to keep RNG tensors CPU; move momentum.
            dev = next(net.parameters()).device
            for state in self.opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v): state[k] = v.to(dev)
            self.sched.load_state_dict(ck['sched']); self.start=ck['step']; self.best=ck['best']; self.best_state=ck['best_state']; self.log=ck['log']; restore_rng(ck['rng'])
            self.pending_validation=ck.get('pending_validation'); self.pending_loss=ck.get('pending_loss')
            print(f'resume {key} at {self.start}/{a.steps}', flush=True)
    def save(self, step):
        self.bundle['jobs'][self.key] = dict(net=self.net.state_dict(), opt=self.opt.state_dict(), sched=self.sched.state_dict(),
                      step=step, best=self.best, best_state=self.best_state, log=self.log, rng=rng_state(),
                      pending_validation=self.pending_validation,pending_loss=self.pending_loss)
        atomic_save(self.bundle, self.path)
    def validate_record(self,step,loss,validate):
        self.pending_validation=step; self.pending_loss=loss; self.net.eval()
        try:
            with torch.no_grad(): val=float(validate())
        except SystemExit:
            self.save(step); raise
        if np.isfinite(val) and val<self.best:
            self.best=val; self.best_state={k:v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
        rec=dict(job=self.key,step=step,loss=loss,val_loss=val)
        rec.update(self.validation_details)
        self.log.append(rec); print(json.dumps(rec),flush=True)
        self.pending_validation=None; self.pending_loss=None; self.save(step)
    def run(self,loss_fn,validate,select_best=True):
        if self.pending_validation is not None:
            self.validate_record(self.pending_validation,self.pending_loss,validate)
        for step in range(self.start,self.a.steps):
            if STOP: self.save(step); raise SystemExit(143)
            self.net.train(); loss=loss_fn(); self.opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(),1); self.opt.step(); self.sched.step()
            interval=10 if self.a.smoke else self.a.eval_every
            if (step+1)%interval==0 or step+1==self.a.steps:
                self.validate_record(step+1,float(loss.detach()),validate)
            if STOP:
                self.save(step+1); raise SystemExit(143)
        if self.best_state is None: raise RuntimeError('no finite validation score')
        if select_best: self.net.load_state_dict(self.best_state)
        self.net.eval()
        return self.log


def sample(rows, n): return np.random.choice(rows, min(n,len(rows)), replace=False)


def export(a, name, net, extra=None):
    atomic_save(dict(state=net.state_dict(), code_signature=CODE_SIGNATURE, args=vars(a), **(extra or {})), Path(a.out)/name)


def load_spaces(a, dev):
    ck = torch.load(Path(a.space_dir or a.out)/'lf_spaces.pt', map_location='cpu', weights_only=False)
    assert ck['args']['smoke'] == a.smoke and ck['code_signature']==CODE_SIGNATURE
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
        return base + mse_mask(pred,y,m) + mse_mask(net.decode(zm),y,m)
    val=np.random.default_rng(54).permutation(d.val)[:min(len(d.val),2048)]
    logs=t.run(lambda:loss(sample(d.train,a.lf_bs),True),lambda:loss(val,False))
    export(a,'lf_spaces.pt',net,dict(mu=d.mu,sd=d.sd,audit=d.audit,log=logs))
    fit_reference(a,d,dev)


def ridge_fit(d, rows, dev, mu, sd, dims=None, y=None, m=None, max_rows=60000):
    y = d.y if y is None else y; m=d.m if m is None else m
    p=1024 if dims is None else dims[1]-dims[0]
    wt=np.zeros((p+1,13),np.float32)
    for j in range(13):
        check_stop()
        r=rows[m[rows,j]]; r=np.random.default_rng(j).permutation(r)[:max_rows]
        if not len(r): continue
        aa=torch.zeros((p+1,p+1),device=dev,dtype=torch.float64); ab=torch.zeros(p+1,device=dev,dtype=torch.float64)
        for i in range(0,len(r),4096):
            check_stop()
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
    for s in (CODE_SIGNATURE,a.targets,a.lf_marker,a.lf_autofl,'K019-batch1-group8-masked-ridge10-v1'): h.update(s.encode())
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
            return (zg-zl).square().sum(-1).mean()+a.w_sig*net['sig'](zg,zl)
        zl=F.normalize(net['P2'](x),dim=-1)
        return net['sig'](zg,zl)
    def val():
        total=0.; n=0
        for i in range(0,len(d.gs_val),a.bs):
            check_stop()
            r=d.gs_val[i:i+a.bs]; total+=float(loss(r,False))*len(r); n+=len(r)
        return total/max(n,1)
    logs=t.run(lambda:loss(sample(d.gs_train,a.bs),True),val)
    export(a,'gs_encoder.pt',net,dict(scale=scale,mu=d.mu,sd=d.sd,log=logs,audit=d.audit))


def load_gs(a,dev):
    ck=torch.load(Path(a.gs_dir or a.out)/'gs_encoder.pt',map_location='cpu',weights_only=False)
    assert ck['code_signature']==CODE_SIGNATURE
    if a.arm=='A3': net=nn.ModuleDict({'gs':GS(a,False),'sig':SigLIP()}).to(dev)
    else: net=nn.ModuleDict({'gs':GS(a,False),'P2':mlp(1024,a.dim),'sig':SigLIP()}).to(dev)
    net.load_state_dict(ck['state']); return frozen(net), ck


def check_stop():
    if STOP: raise SystemExit(143)


@torch.no_grad()
def cache_gs_pairs(a,d,dev):
    """Only source split 0/1; test slide is never cached for flow training."""
    gs,ck=load_gs(a,dev)
    path=Path(a.out)/'pairs.npz'; ckpath=Path(a.out)/'ckpt.pt'
    state=dict(code_signature=CODE_SIGNATURE,args=vars(a),cache={})
    if ckpath.exists():
        state=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert state['code_signature']==CODE_SIGNATURE and state['args']==vars(a)
    for split,rows in [('train',d.gs_train),('val',d.gs_val)]:
        rec=state['cache'].setdefault(split,dict(rows=rows,done=0,parts=[]))
        assert np.array_equal(rec['rows'],rows)
        for i in range(rec['done'],len(rows),a.bs):
            rr=rows[i:i+a.bs]
            rec['parts'].append(gs['gs'](prep(d.images(rr),ck['scale'],dev,False,a.smoke)).cpu().numpy())
            rec['done']=i+len(rr)
            if STOP or rec['done']==len(rows) or (i//a.bs+1)%20==0: atomic_save(state,ckpath)
            check_stop()
    np.savez(path,train_rows=d.gs_train,val_rows=d.gs_val,
        train_z=np.concatenate(state['cache']['train']['parts']),val_z=np.concatenate(state['cache']['val']['parts']),
        code_signature=CODE_SIGNATURE)


@torch.no_grad()
def sinkhorn_log(cost,epsilon,iterations=100):
    """Uniform balanced entropic OT, log-domain scaling."""
    n,m=cost.shape; logk=-cost/epsilon
    u=cost.new_zeros(n); v=cost.new_zeros(m)
    for _ in range(iterations):
        u=-math.log(n)-torch.logsumexp(logk+v[None,:],1)
        v=-math.log(m)-torch.logsumexp(logk+u[:,None],0)
    return torch.exp(logk+u[:,None]+v[None,:])


@torch.no_grad()
def entropic_gw(x,y,epsilon=.005,iterations=40):
    """Squared-loss entropic GW (Peyre/Cuturi/Solomon 2016, Algorithm 1).
    Only intra-space cosine distances enter. Distance-profile initialization
    breaks the uninformative uniform stationary point without cross-frame costs.
    Equal-size minibatches; uniform marginals; no identity/label input.
    """
    assert len(x)==len(y)
    x=F.normalize(x.double(),dim=-1); y=F.normalize(y.double(),dim=-1)
    c=(1-x@x.T).clamp_min(0); d=(1-y@y.T).clamp_min(0)
    c.fill_diagonal_(0); d.fill_diagonal_(0)
    # Quantile distance signatures are invariant to rotations and permutations.
    pc=c.sort(1).values; pd=d.sort(1).values
    init=(pc[:,None,:]-pd[None,:,:]).square().mean(-1)
    coupling=sinkhorn_log(init,max(float(init.mean())*.05,1e-8))
    n=len(x); constant=c.square().mean(1)[:,None]+d.square().mean(1)[None,:]
    for _ in range(iterations):
        check_stop()
        cost=2*(constant-2*c@coupling@d.T)
        updated=sinkhorn_log(cost,epsilon)
        change=(updated-coupling).abs().max()
        coupling=updated
        if change<1e-9: break
    # Round to exact balanced marginals (Altschuler et al. rounding).
    coupling*=torch.minimum(1/coupling.sum(1).clamp_min(1e-30)/n,torch.ones(n,device=x.device))[:,None]
    coupling*=torch.minimum(1/coupling.sum(0).clamp_min(1e-30)/n,torch.ones(n,device=x.device))[None,:]
    a=(1/n-coupling.sum(1)).clamp_min(0); b=(1/n-coupling.sum(0)).clamp_min(0)
    if a.sum()>1e-15: coupling+=a[:,None]*b[None,:]/a.sum()
    return coupling


def map_partition(d):
    """Fixed 80/20 source split-1 partition, grouped by slide/FOV."""
    rows=d.gs_val
    groups=np.stack([d.slide[rows],d.fov[rows]],1)
    unique,inv=np.unique(groups,axis=0,return_inverse=True)
    assert len(unique)>=2,'need two source split-1 FOVs'
    chosen=np.random.default_rng(5401).permutation(len(unique))[:max(1,math.ceil(.2*len(unique)))]
    hold=np.isin(inv,chosen)
    return rows[~hold],rows[hold]


def train_flow(a,d,dev):
    spaces=load_spaces(a,dev); gs,_=load_gs(a,dev); net=Velocity(a.dim).to(dev)
    if a.init_flow:
        initial=torch.load(a.init_flow,map_location='cpu',weights_only=False)
        assert initial['code_signature']==CODE_SIGNATURE
        net.load_state_dict(initial['state'])
    pairs=None
    if a.map_source!='lf':
        pairs=dict(np.load(a.pair_cache)); assert str(pairs['code_signature'])==CODE_SIGNATURE
        assert np.array_equal(pairs['train_rows'],d.gs_train) and np.array_equal(pairs['val_rows'],d.gs_val)
    if a.map_source=='split1':
        assert a.arm in ('A1-GS','direct') and a.sigma_bridge==0 and a.init_flow
        source,valid=map_partition(d)
    else:
        source=d.train if a.map_source=='lf' else d.gs_train
        valid=d.val if a.map_source=='lf' else d.gs_val
    t=Trainer(a,'flow',net,a.flow_lr); endpoint_audit=[]
    def endpoints(rows,target_rows=None):
        with torch.no_grad():
            if a.map_source=='lf': z2=F.normalize(gs['P2'](d.x(rows,dev)),dim=-1)
            else:
                key='val' if a.map_source=='split1' or np.isin(rows,d.gs_val).all() else 'train'
                r=pairs[key+'_rows']; idx=np.searchsorted(r,rows); assert np.array_equal(r[idx],rows)
                z2=torch.as_tensor(pairs[key+'_z'][idx],device=dev)
            if a.arm in ('A1-LF','A1-GS','direct'): z1=spaces.lf(d.x(rows,dev))
            else:
                assert (d.hist[target_rows]<0).all() and not np.intersect1d(rows,target_rows).size
                if a.arm=='A2': z1=spaces.lf(d.x(target_rows,dev))
                else:
                    y,m=d.labels(target_rows,dev); z1=spaces.marker(y,m)
                if a.arm=='A2' and a.a2_cost=='gw':
                    coupling=entropic_gw(z2,z1)
                    # Sample the joint coupling, retaining both sampled endpoints.
                    ij=torch.multinomial(coupling.flatten(),len(rows),replacement=True)
                    ii,jj=ij//len(z1),ij%len(z1)
                else:
                    from scipy.optimize import linear_sum_assignment
                    ii,jj=linear_sum_assignment(torch.cdist(z2,z1).square().cpu().numpy())
                    ii=torch.as_tensor(ii,device=dev); jj=torch.as_tensor(jj,device=dev)
                collisions=int((np.asarray(rows)[ii.cpu().numpy()]==np.asarray(target_rows)[jj.cpu().numpy()]).sum())
                assert not collisions
                endpoint_audit.append(dict(source_gs=int((d.hist[rows]>=0).sum()),target_gs=int((d.hist[target_rows]>=0).sum()),
                    same_cell_assignments=collisions,n=len(rows),cost=a.a2_cost if a.arm=='A2' else 'euclidean'))
                z2,z1=z2[ii],z1[jj]
        return z2,z1
    n=min(a.lf_bs,len(valid),len(d.val))
    vr=np.random.default_rng(54).permutation(valid)[:n]; vt=np.random.default_rng(55).permutation(d.val)[:n]
    def objective(rows,target_rows=None,validation=False):
        # Validation GW sampling/bridge noise must not consume training RNG.
        saved=rng_state() if validation else None
        if validation: seed_all(54)
        z2,z1=endpoints(rows,target_rows)
        if a.arm=='direct': value=(F.normalize(net(z2,z2.new_zeros(len(z2))),dim=-1)-z1).square().sum(-1).mean()
        else:
            tt=torch.rand(len(z2),1,device=dev); eps=torch.randn_like(z1)
            xt=(1-tt)*z2+tt*z1+a.sigma_bridge*tt*(1-tt)*eps
            velocity=z1-z2+a.sigma_bridge*(1-2*tt)*eps
            value=F.mse_loss(net(xt,tt),velocity)
        if saved is not None: restore_rng(saved)
        return value
    def update():
        n=min(a.lf_bs,len(source),len(d.train))
        return objective(sample(source,n),sample(d.train,n))
    fixed=a.arm in ('A1-GS','direct') and a.map_source!='lf'
    def validate():
        if not fixed: return objective(vr,vt,True)
        preds=[]; truths=[]; cosines=[]; losses=[]
        with torch.no_grad():
            for i in range(0,len(valid),a.lf_bs):
                check_stop(); rr=valid[i:i+a.lf_bs]; losses.append((float(objective(rr,validation=True)),len(rr)))
                z2,z1=endpoints(rr)
                mapped=F.normalize(net(z2,z2.new_zeros(len(z2))),dim=-1) if a.arm=='direct' else integrate(net,z2,a.euler_steps)
                preds.append(spaces.decode(mapped)[:,12].cpu().numpy()); truths.append(d.y[rr,12])
                cosines.append((mapped*z1).sum(1).cpu().numpy())
        t.validation_details=dict(holdout_n=len(valid),holdout_autofl_r=corr(np.concatenate(preds),np.concatenate(truths)),
            holdout_mapped_cosine=float(np.concatenate(cosines).mean()),informational_only=True)
        return sum(loss*n for loss,n in losses)/len(valid)
    # Preserve the initializer's pre-curve across interrupted fine tuning.
    if 'pre_finetune_curve' not in t.bundle:
        with torch.no_grad(): pre_loss=float(validate())
        t.bundle['pre_finetune_curve']=dict(step=0,val_loss=pre_loss,**t.validation_details)
    logs=t.run(update,validate,select_best=not fixed)
    export(a,'flow.pt',net,dict(log=logs,pre_finetune_curve=t.bundle['pre_finetune_curve'],source_rows=source,validation_rows=valid,
        target_rows=d.train,selection='fixed final iterate' if fixed else 'minimum validation loss',
        endpoint_audit=endpoint_audit,initial_flow=a.init_flow,paired=a.arm in ('A1-LF','A1-GS','direct'),
        same_cell_matches_forbidden=a.arm in ('A2','A2-EM'),capacity=sum(p.numel() for p in net.parameters())))
    if fixed:
        json_write(Path(a.out)/'finetune_curve.json',dict(curve=[t.bundle['pre_finetune_curve']]+logs,
            source_rows=source,holdout_rows=valid,grouping='slide,FOV',selection='none; fixed final iterate',
            holdout_fraction_fovs=math.ceil(.2*len(np.unique(d.fov[d.gs_val])))/len(np.unique(d.fov[d.gs_val]))))
    if a.arm=='A2' and a.a2_cost=='gw':
        sanity=[]
        with torch.no_grad():
            for rep in range(5):
                rr=np.random.default_rng(540+rep).choice(d.val,n,replace=False); x=d.x(rr,dev)
                z2=F.normalize(gs['P2'](x),dim=-1); z1=spaces.lf(x); coupling=entropic_gw(z2,z1)
                sanity.append(dict(n=n,identity_rate=float((coupling.argmax(1)==torch.arange(n,device=dev)).double().mean()),
                    identity_mass=float(coupling.diag().sum()),chance=1/n))
        json_write(Path(a.out)/'gw_identity_sanity.json',dict(source='P2(f)',target='P1(f)',same_lf_cells=True,rows=sanity,
            mean_identity_rate=float(np.mean([r['identity_rate'] for r in sanity])),chance=1/n))

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
            if a.arm in ('A1-LF','A1-GS','A2','A2-EM','direct'):
                self.flow=Velocity(a.dim).to(dev)
                self.flow.load_state_dict(torch.load(Path(a.out)/'flow.pt',map_location='cpu',weights_only=False)['state']); frozen(self.flow)
    @torch.no_grad()
    def images(self,x):
        z=self.base['gs'](x) if self.a.arm=='A0' else self.gs['gs'](x)
        if self.a.arm=='A0': p=self.base['S'](z*math.sqrt(self.a.dim)); mapped=z
        else:
            mapped=(F.normalize(self.flow(z,z.new_zeros(len(z))),dim=-1) if self.a.arm=='direct' else integrate(self.flow,z,self.a.euler_steps)) if self.a.arm!='A3' else z
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
            check_stop()
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
        check_stop()
        sim=a[s:s+512]@b.T; diag=sim[np.arange(len(sim)),np.arange(s,s+len(sim))]
        # Ties count pessimistically, matching train_gs_bind.py.
        ranks.extend((sim>=diag[:,None]).sum(1)-1)
    ranks=np.asarray(ranks)
    return dict(n=len(a),R_at_1=float((ranks<1).mean()),R_at_10=float((ranks<10).mean()),
                median_rank=float(np.median(ranks+1)),chance_R_at_1=1/len(a),chance_R_at_10=min(10/len(a),1))


def run_eval(a,d,dev):
    fit_reference(a,d,dev); ref=dict(np.load(a.ridge_cache or (Path(a.out)/'ridge.npz')))
    p=Predictor(a,d,dev); rows=(map_partition(d)[1] if a.map_source=='split1' and a.arm in ('A1-GS','direct') else d.gs_val) if a.validation_only else d.gs_test
    output='validation' if a.validation_only else 'eval'
    metrics=[]
    path=Path(a.out)/'ckpt.pt'
    bundle=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {}
    fingerprints=[]
    files=([Path(a.baseline or Path(a.data)/f'runs/n029_tt6c_x_{a.holdout}/gs_bind.pt')] if a.arm=='A0' else
           [Path(a.out)/f for f in ('lf_spaces.pt','gs_encoder.pt','flow.pt')])
    for file in files:
        if file.exists(): fingerprints.append((str(file),file.stat().st_size,file.stat().st_mtime_ns))
    signature=(CODE_SIGNATURE,fingerprints,rows.tolist(),a.smoke,a.euler_steps)
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
            if a.arm in ('A1-LF','A1-GS','A2','A2-EM','direct'): zlf1.append(p.spaces.lf(d.x(rr,dev)).cpu().numpy())
            progress['done']=s+len(rr)
            if STOP or progress['done']==len(rows) or (s//a.bs+1)%20==0:
                bundle['evaluations'][output]=progress; atomic_save(bundle,path)
            if STOP: raise SystemExit(143)
    pred,zg,mapped,own,zl,teacher=[np.concatenate(v) for v in (gs_pred,zgs,zmapped,lf_pred,zlf,fixed)]
    j=12; observed=d.m[rows,j]
    result=dict(arm=a.arm,holdout=a.holdout,map_source=a.map_source,a2_cost=a.a2_cost,smoke=a.smoke,validation_only=a.validation_only,
                M1_autofl_truth_r=corr(pred[observed,j],d.y[rows[observed],j]),
                M3=retrieval(zg,zl),M4_GS_LF=geometry(zg,zl),audit=d.audit)
    if a.arm in ('A1-LF','A1-GS','A2','A2-EM','direct'):
        result['M4_mapped_GS_LFspace1']=geometry(mapped,np.concatenate(zlf1))
        result['M3_mapped_GS_LFspace1']=retrieval(mapped,np.concatenate(zlf1))
    if a.arm=='A3':
        c=(zg*zl).sum(1)
        result['modality_gap']=dict(z_L_definition='P_L(f_own)',cos_zG_zL_mean=float(c.mean()),cos_zG_zL_sd=float(c.std()),
            distance_zG_zL_mean=float(np.linalg.norm(zg-zl,axis=1).mean()))
    if a.arm not in ('A0','A3'):
        z1own=np.concatenate(zlf1)
        native_cos=(zg*zl).sum(1)
        cross_cos=(zg*z1own).sum(1)
        result['modality_gap']=dict(z_L_definition='P2(f_own); native LF latent paired to E_G',
            cos_zG_P2_own_mean=float(native_cos.mean()),cos_zG_P2_own_sd=float(native_cos.std()),
            cos_zG_zL_mean=float(native_cos.mean()),cos_zG_zL_sd=float(native_cos.std()),
            cos_zG_P1_own_mean=float(cross_cos.mean()),cos_zG_P1_own_sd=float(cross_cos.std()),
            distance_zG_P2_own_mean=float(np.linalg.norm(zg-zl,axis=1).mean()))
        distances=[]
        for k in range(0,len(zl),512):
            check_stop(); sim=zl[k:k+512]@zl.T
            sim[np.arange(len(sim)),np.arange(k,k+len(sim))]=-np.inf
            distances.extend(np.sqrt(np.maximum(2-2*sim.max(1),0)))
        result['modality_gap']['distance_P2_nearest_other_mean']=float(np.mean(distances))
        if a.arm=='A1-LF':
            with torch.no_grad():
                oracle_z=integrate(p.flow,torch.as_tensor(zl,device=dev),a.euler_steps)
                oracle_pred=p.spaces.decode(oracle_z).cpu().numpy()
            result['oracle_A1_LF']=dict(M1_autofl_truth_r=corr(oracle_pred[observed,12],d.y[rows[observed],12]),
                start='P2(f_own); diagnostic requires own LF, not deployable GS inference')
            np.savez(Path(a.out)/f'{output}_oracle.npz',rows=rows,pred=oracle_pred,z_start=zl)
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
        if a.arm in ('A1-LF','A1-GS','A2','A2-EM','direct'):
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
        self.plain=mlp(1050,13)
        self.labels_em=mlp(26,dim); self.labels_head=mlp(2*dim,13)
    def forward(self,x,y,m):
        z=self.spaces.lf(x); em=self.spaces.marker(y,m)
        le=F.normalize(self.labels_em(torch.cat([y*m,m.float()],-1)),dim=-1)
        return (self.spaces.decode(z),self.fused(torch.cat([z,em],-1)*math.sqrt(self.spaces.dim)),
                self.labels_head(torch.cat([torch.zeros_like(le),le],-1)*math.sqrt(self.spaces.dim)),
                self.plain(torch.cat([x,y*m,m.float()],-1)))


def completion_input(d,rows,typ,j):
    m=d.m[rows].copy()
    if typ=='C1': m[:,8:]=False
    else: m[:,:8]=False; m[:,12]=False
    m[:,j]=False
    assert not m[:,j].any()
    return np.concatenate([d.y[rows]*m,m.astype(np.float32)],1),m


def completion_linear(d,train,rows,typ,j,stats,dev):
    """Target-specific ridge controls. Only source split-0; deterministic cap.
    Closed-form loops checkpoint by exiting promptly; cheap solves recompute.
    """
    observed=d.m[train,j]
    eligible=(d.m[train,:8].sum(1)==1) if typ=='C1' else d.m[train,8:12].any(1)
    tr=train[observed & eligible]
    tr=np.random.default_rng(54+j).permutation(tr)[:60000]
    if not len(tr): return np.full(len(rows),np.nan),np.full(len(rows),np.nan),{},0
    controls=[]; weights={}
    for include_lf in (False,True):
        p=1050 if include_lf else 26
        aa=torch.zeros(p+1,p+1,device=dev,dtype=torch.float64)
        ab=torch.zeros(p+1,device=dev,dtype=torch.float64)
        def features(rr):
            lab,_=completion_input(d,rr,typ,j)
            x=np.concatenate([(d.raw(rr)-stats[0])/stats[1],lab],1) if include_lf else lab
            return torch.as_tensor(np.c_[x,np.ones(len(x),np.float32)],device=dev,dtype=torch.float64)
        for k in range(0,len(tr),4096):
            check_stop(); rr=tr[k:k+4096]; x=features(rr)
            aa+=x.T@x; ab+=x.T@torch.as_tensor(d.y[rr,j],device=dev,dtype=torch.float64)
        reg=torch.eye(p+1,device=dev,dtype=torch.float64)*10; reg[-1,-1]=0
        check_stop(); wt=torch.linalg.solve(aa+reg,ab)
        weights['f_plus_labels' if include_lf else 'labels_linear']=wt.cpu().numpy()
        pred=[]
        for k in range(0,len(rows),4096):
            check_stop(); pred.append((features(rows[k:k+4096])@wt).cpu().numpy())
        controls.append(np.concatenate(pred))
    weights['train_rows']=tr
    return controls[0],controls[1],weights,len(tr)


def completion_stage(a,d,dev):
    assert a.targets=='targets_v5cx_v10.npz'
    dest=Path(a.out)/'completion'; dest.mkdir(exist_ok=True)
    report=[]; coherence=[]; mask_audit=[]
    for protocol,trainb,testb in [('fov',None,None),('draw0_to_1',0,1),('draw1_to_0',1,0)]:
        train=d.rows(0,trainb); val=d.rows(1,trainb); test=d.rows(2,testb)
        assert np.all(d.hist[train]<0) and np.all(d.split[train]==0)
        if trainb is not None: assert np.all(d.batch[train]==trainb) and not np.intersect1d(train,test).size
        stats=d.stats(train); seed_all(a.seed+{'fov':100,'draw0_to_1':200,'draw1_to_0':300}[protocol])
        net=Completion(a.dim).to(dev); trainer=Trainer(a,'completion_'+protocol,net,a.lf_lr)
        def objective(rows,validation=False):
            x=d.x(rows,dev,stats); y,m=d.labels(rows,dev)
            if validation:
                gen=torch.Generator(device=dev).manual_seed(54)
                keep=(torch.rand(m.shape,device=dev,generator=gen)>.5)&m
            else: keep=(torch.rand(m.shape,device=dev)>.5)&m
            hidden=m & ~keep
            pl,pf,pc,pplain=net(x,y,keep)
            loss=mse_mask(pl,y,hidden)+mse_mask(pf,y,hidden)+mse_mask(pc,y,hidden)+mse_mask(pplain,y,hidden)
            if not validation:
                z=net.spaces.lf(x); em=net.spaces.marker(y,m)
                loss=loss+net.spaces.sig(z,em,near_duplicate(y,m,a.tau))
            else:
                mask_audit.append(dict(protocol=protocol,validation_observed=int(m.sum()),
                    validation_scored=int(hidden.sum()),scored_visible=int((hidden&keep).sum())))
            return loss
        vr=np.random.default_rng(54).permutation(val)[:min(1024,len(val))]
        logs=trainer.run(lambda:objective(sample(train,a.lf_bs)),lambda:objective(vr,True))
        atomic_save(dict(state=net.state_dict(),code_signature=CODE_SIGNATURE,mu=stats[0],sd=stats[1],
            train_rows=train,val_rows=val,test_rows=test,protocol=protocol,args=vars(a),log=logs),dest/f'{protocol}.pt')
        net.eval()
        single=d.m[test,:8].sum(1)==1; pooled=d.m[test,8:12].any(1)
        tasks=[('C1',12,test[single & d.m[test,12]])]+[('C2',j,test[pooled & d.m[test,j]]) for j in range(8,12)]
        for typ,j,rows in tasks:
            if not len(rows):
                report.append(dict(test=typ,protocol=protocol,target=d.names[j],n=0,status='no observed test truth after QC'))
                continue
            parts=[[],[],[],[]]; inputs=[]
            for k in range(0,len(rows),a.lf_bs):
                check_stop(); rr=rows[k:k+a.lf_bs]; _,mi=completion_input(d,rr,typ,j)
                y,_=d.labels(rr,dev)
                with torch.no_grad(): pred=net(d.x(rr,dev,stats),y,torch.as_tensor(mi,device=dev))
                for arr,prediction in zip(parts,pred): arr.append(prediction.cpu().numpy())
                inputs.append(mi)
            pl,pf,pc,plain=[np.concatenate(v) for v in parts]; ins=np.concatenate(inputs)
            # Stacking weights use source split-1 only, with the exact task mask.
            eligible=(d.m[val,:8].sum(1)==1) if typ=='C1' else d.m[val,8:12].any(1)
            sr=val[eligible & d.m[val,j]]; stack_parts=[]
            for k in range(0,len(sr),a.lf_bs):
                check_stop(); rr=sr[k:k+a.lf_bs]; _,mi=completion_input(d,rr,typ,j); yy,_=d.labels(rr,dev)
                with torch.no_grad(): q=net(d.x(rr,dev,stats),yy,torch.as_tensor(mi,device=dev))
                stack_parts.append(np.c_[q[0][:,j].cpu().numpy(),q[2][:,j].cpu().numpy(),np.ones(len(rr))])
            sx=np.concatenate(stack_parts) if len(sr) else np.empty((0,3))
            groups=[('all',np.ones(len(rows),bool))]
            if typ=='C1': groups += [(d.names[k],d.m[rows,k]) for k in range(8)]
            stack_weights=[]; stack_fit_masks=[]; stack_predictions={}; stack_counts={}
            for marker,sel in groups:
                fit=np.ones(len(sr),bool) if marker=='all' else d.m[sr,d.names.index(marker)]
                sw=np.linalg.lstsq(sx[fit],d.y[sr[fit],j],rcond=None)[0] if fit.any() else np.full(3,np.nan)
                stack_predictions[marker]=np.c_[pl[:,j],pc[:,j],np.ones(len(rows))]@sw
                stack_counts[marker]=int(fit.sum()); stack_weights.append(sw); stack_fit_masks.append(fit)
            stacked_all=stack_predictions['all']; stacked=stacked_all.copy()
            if typ=='C1':
                for marker,sel in groups[1:]: stacked[sel]=stack_predictions[marker][sel]
            np.savez(dest/f'{protocol}_{typ}_{j}_stack.npz',weight=np.asarray(stack_weights),train_rows=sr,
                group_names=np.asarray([g[0] for g in groups]),group_fit_mask=np.asarray(stack_fit_masks),split=d.split[sr],batch=d.batch[sr])
            linear,combined,weights,nt=completion_linear(d,train,rows,typ,j,stats,dev)
            np.savez(dest/f'{protocol}_{typ}_{j}_ridge.npz',**weights)
            groups=[('all',np.ones(len(rows),bool))]
            if typ=='C1': groups += [(d.names[k],d.m[rows,k]) for k in range(8)]
            for marker,sel in groups:
                truth=d.y[rows[sel],j]
                scores={key:(corr(pred[sel],truth) if nt else None) for key,pred in
                    [('r_lf',pl[:,j]),('r_fused',pf[:,j]),('r_labels_mlp',pc[:,j]),
                     ('r_labels_linear',linear),('r_f_plus_labels_ridge',combined),('r_stacked',stack_predictions[marker]),('r_plain_mlp',plain[:,j])]}
                controls=[scores[k] for k in scores if k!='r_fused']
                complete=all(v is not None for v in scores.values())
                delta=scores['r_fused']-max(controls) if complete else None
                label_rs=[scores[k] for k in ('r_labels_linear','r_labels_mlp') if scores[k] is not None]
                report.append(dict(test=typ,protocol=protocol,target=d.names[j],observed_marker=marker,
                    n=int(sel.sum()),n_train_target=nt,**scores,
                    gain_over_all_controls=delta,
                    gain_vs_stacked=scores['r_fused']-scores['r_stacked'] if scores['r_fused'] is not None and scores['r_stacked'] is not None else None,
                    gain_vs_plain_mlp=scores['r_fused']-scores['r_plain_mlp'] if scores['r_fused'] is not None and scores['r_plain_mlp'] is not None else None,
                    n_stack_train=stack_counts[marker],
                    gain_vs_lf=product(1,scores['r_fused']-scores['r_lf']) if scores['r_fused'] is not None and scores['r_lf'] is not None else None,
                    gain_vs_labels_linear=scores['r_fused']-scores['r_labels_linear'] if scores['r_fused'] is not None and scores['r_labels_linear'] is not None else None,
                    gain_vs_labels_mlp=scores['r_fused']-scores['r_labels_mlp'] if scores['r_fused'] is not None and scores['r_labels_mlp'] is not None else None,
                    gain_vs_f_plus_labels_ridge=scores['r_fused']-scores['r_f_plus_labels_ridge'] if scores['r_fused'] is not None and scores['r_f_plus_labels_ridge'] is not None else None,
                    hidden_information_supported=bool(delta is not None and delta>0),
                    labels_only_predicts_autofl=bool(typ=='C1' and label_rs and max(abs(v) for v in label_rs)>=.2),
                    flag_rule='abs(labels-only r)>=0.2 descriptive crosstalk warning; not causal evidence',
                    mean_observed_inputs=float(ins[sel].sum(1).mean()) if sel.any() else None,
                    hidden_target_in_input=int(ins[sel,j].sum()),
                    status='ok' if nt else 'unavailable: no source training truth'))
            np.savez(dest/f'{protocol}_{typ}_{j}.npz',rows=rows,lf=pl,fused=pf,labels_mlp=pc,
                labels_linear=linear,f_plus_labels_ridge=combined,stacked=stacked,stacked_all=stacked_all,plain_mlp=plain,truth=d.y[rows],mask=d.m[rows],
                input_mask=ins,names=d.names)
        pop=test[single]; lfs=[]; fus=[]
        for k in range(0,len(pop),a.lf_bs):
            check_stop(); rr=pop[k:k+a.lf_bs]; _,mi=completion_input(d,rr,'C1',12); y,_=d.labels(rr,dev)
            with torch.no_grad(): l,f,_,_=net(d.x(rr,dev,stats),y,torch.as_tensor(mi,device=dev))
            lfs.append(l.cpu().numpy()); fus.append(f.cpu().numpy())
        pl=np.concatenate(lfs); pf=np.concatenate(fus)
        np.savez(dest/f'{protocol}_C3.npz',rows=pop,lf=pl,fused=pf,observed_values=d.y[pop],mask=d.m[pop],names=d.names)
        for marker in ('single:CD19','single:CD3','single:CD14','single:CD56'):
            j=d.names.index(marker); obs=d.m[pop,j]
            if not obs.any(): continue
            cutoff=float(np.quantile(d.y[pop[obs],j],.95)); hi=obs&(d.y[pop,j]>=cutoff)
            for k,name in enumerate(d.names):
                if k==j: continue
                coherence.append(dict(protocol=protocol,conditioning_marker=marker,decoded_target=name,n_high=int(hi.sum()),
                    observed_threshold=cutoff,lf_high_mean=float(pl[hi,k].mean()),fused_high_mean=float(pf[hi,k].mean()),
                    lf_population_mean=float(pl[:,k].mean()),fused_population_mean=float(pf[:,k].mean()),
                    status='descriptive; other-marker truth absent'))
        print(f'completion {protocol}: trained={len(train)}, tested={len(test)}',flush=True)
    csv_write(dest/'completion.csv',report); csv_write(dest/'coherence.csv',coherence)
    json_write(dest/'completion.json',dict(metrics=report,coherence=coherence,audit=d.audit,validation_masks=mask_audit,
        claim_rule='r_fused > max(r_lf, r_labels_linear, r_labels_mlp, r_f_plus_labels_ridge, r_stacked, r_plain_mlp); descriptive, no significance test'))

def artifact(path):
    path=Path(path).resolve(); st=path.stat()
    return dict(path=str(path),size=st.st_size,mtime_ns=st.st_mtime_ns)


def signature(a,inputs=()):
    payload=dict(code_signature=CODE_SIGNATURE,args=vars(a),inputs=[artifact(p) for p in inputs])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for stream in self.streams: stream.write(s); stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams: stream.flush()


def run_stage(a,d,dev,inputs=(),outputs=()):
    """One immutable completion marker per stage; in-progress work uses ckpt.pt."""
    check_stop(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    mark=out/(a.stage+'.done.json'); sig=signature(a,inputs)
    if mark.exists():
        prior=json.loads(mark.read_text())
        if prior['signature']!=sig: raise ValueError(f'stage signature mismatch: {mark}; use a new versioned output directory')
        if all(Path(p).exists() for p in outputs) and prior['outputs']==[artifact(p) for p in outputs]:
            print(f'SKIP {a.stage} {out}',flush=True); return
    seed_all(a.seed)
    with (out/(a.stage+'.log')).open('a',buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout,logfile)),contextlib.redirect_stderr(Tee(sys.stderr,logfile)):
            print(json.dumps(dict(event='start',stage=a.stage,arm=a.arm,holdout=a.holdout,code_signature=CODE_SIGNATURE)),flush=True)
            began=time.time()
            json_write(out/'data_audit.json',d.audit)
            if a.stage=='clinical':
                import importlib.util
                spec=importlib.util.spec_from_file_location('n054_clinical_local',Path(__file__).with_name('n054_clinical.py'))
                clinical=importlib.util.module_from_spec(spec); spec.loader.exec_module(clinical)
                # Same module instance for signal STOP and model classes under python -I.
                clinical.n=sys.modules[__name__]
                clinical.main(['--run',a.out,'--out',str(out/'clinical'),'--arm',a.arm,
                    '--holdout',a.holdout,'--data',a.data,'--device',str(dev),'--threads',str(a.threads),
                    '--batch-size',str(a.bs),'--tta','0' if a.smoke and a.arm=='A0' else '1',*(['--smoke'] if a.smoke else [])])
            else:
                functions=dict(lf_spaces=train_lf,gs_encoder=train_gs,gs_cache=cache_gs_pairs,
                    flow=train_flow,eval=run_eval,validation=run_eval,completion=completion_stage)
                functions[a.stage](a,d,dev)
            check_stop()
            assert all(Path(p).exists() for p in outputs),f'missing exports for {a.stage}'
            json_write(mark,dict(signature=sig,code_signature=CODE_SIGNATURE,args=vars(a),
                outputs=[artifact(p) for p in outputs],seconds=time.time()-began))
            print(json.dumps(dict(event='done',stage=a.stage,seconds=round(time.time()-began,2))),flush=True)


def config(base,**kw):
    values=vars(base).copy(); values.update(kw); return argparse.Namespace(**values)


def freeze_selection(a,candidates):
    """Only source-slide split-1 measured autofl r; no test/clinical fields."""
    families=['A1-GS','A1-LF','A2','A3','direct']; chosen={}; scores=[]
    for cand in candidates:
        result=json.loads((Path(cand.out)/'validation.json').read_text())
        assert result['validation_only'] and result['holdout']==a.holdout
        rows=np.load(Path(cand.out)/'validation_gs_preds.npz')['rows']
        assert np.all(result['audit']['gs_val_splits']==np.array([1]))
        row=dict(arm=cand.arm,sigma_bridge=cand.sigma_bridge,w_sig=cand.w_sig,run=str(Path(cand.out).resolve()),
            map_source=cand.map_source,a2_cost=cand.a2_cost,
            r=result['M1_autofl_truth_r'],validation=artifact(Path(cand.out)/'validation.json'),n=len(rows))
        scores.append(row)
    for arm in families:
        eligible=[r for r in scores if r['arm']==arm and (arm not in ('A1-GS','direct') or r['map_source']=='split1') and
                  (arm!='A2' or r['a2_cost']=='gw') and (arm in ('A1-GS','direct') or (r['r'] is not None and np.isfinite(r['r'])))]
        if not eligible: raise RuntimeError(f'no finite source-validation autofl r for {arm}')
        # Stable predeclared tie break: lower sigma or lower w_sig.
        chosen[arm]=eligible[0] if arm in ('A1-GS','direct') else max(eligible,key=lambda r:(r['r'],-r['sigma_bridge'],-r['w_sig']))
    payload=dict(code_signature=CODE_SIGNATURE,holdout=a.holdout,criterion='A1-GS/direct fixed split1 sigma0 final iterate, no selection; other families source split-1 autofl r',
                 selected=chosen,grid=scores)
    path=Path(a.out)/'selection.json'
    if path.exists():
        if json.loads(path.read_text())!=payload: raise ValueError('frozen selection differs; use a fresh versioned directory')
    else: json_write(path,payload)
    return payload


def driver(a,dev):
    root=Path(a.out); root.mkdir(parents=True,exist_ok=True)
    if a.unit=='lf':
        d=Data(config(a,targets='targets_tt6c.npz'))
        for arm,name in [('A1-LF','space1'),('A3','hub')]:
            c=config(a,stage='lf_spaces',out=str(root/name),arm=arm,targets='targets_tt6c.npz',ridge_cache=str(root/'ridge.npz'))
            run_stage(c,d,dev,outputs=[root/name/'lf_spaces.pt',root/'ridge.npz'])
        c=config(a,stage='completion',out=str(root/'completion'),targets='targets_v5cx_v10.npz',arm='A3')
        dc=Data(c)
        run_stage(c,dc,dev,outputs=[root/'completion/completion/completion.json',root/'completion/completion/completion.csv',
            *[root/'completion/completion'/f'{p}.pt' for p in ('fov','draw0_to_1','draw1_to_0')]])
        return
    a=config(a,holdout=a.unit)
    lf=Path(a.lf_unit); sp=lf/'space1'; hub=lf/'hub'; ridge=lf/'ridge.npz'
    for f in (sp/'lf_spaces.done.json',hub/'lf_spaces.done.json',lf/'completion/completion.done.json'):
        if not f.exists(): raise RuntimeError(f'LF unit incomplete: {f}; wait for its whole preemption chain')
        assert json.loads(f.read_text())['code_signature']==CODE_SIGNATURE,'LF unit code signature differs'
    a=config(a,ridge_cache=str(ridge)); d=Data(a)
    space2=root/'space2'
    c=config(a,stage='gs_encoder',arm='A1-LF',out=str(space2),space_dir=str(sp))
    run_stage(c,d,dev,inputs=[sp/'lf_spaces.pt'],outputs=[space2/'gs_encoder.pt'])
    c=config(c,stage='gs_cache',out=str(space2/'source_cache'),gs_dir=str(space2))
    run_stage(c,d,dev,inputs=[space2/'gs_encoder.pt'],outputs=[space2/'source_cache/pairs.npz'])
    candidates=[]
    # Hub GS encoders trained once for each weight, without bridge-dependent copies.
    for weight in (0,.5):
        out=root/f'A3_w{weight:g}'
        c=config(a,stage='gs_encoder',arm='A3',w_sig=weight,out=str(out),space_dir=str(hub))
        run_stage(c,d,dev,inputs=[hub/'lf_spaces.pt'],outputs=[out/'gs_encoder.pt'])
        candidates.append(config(c,gs_dir=str(out)))
    def fit_map(name,arm,sigma=0,map_source='split0',init='',a2_cost='gw',small=False):
        out=root/name
        c=config(a,stage='flow',arm=arm,out=str(out),sigma_bridge=sigma,space_dir=str(sp),gs_dir=str(space2),
            pair_cache=str(space2/'source_cache/pairs.npz'),init_flow=str(init),map_source=map_source,a2_cost=a2_cost,
            steps=min(a.steps,20) if a.smoke else (a.gs_map_steps if small else a.steps))
        deps=[sp/'lf_spaces.pt',space2/'gs_encoder.pt',space2/'source_cache/pairs.npz']
        if init: deps.append(Path(init))
        run_stage(c,d,dev,inputs=deps,outputs=[out/'flow.pt'])
        candidates.append(c)
    for sigma in (0,.1):
        initial=root/f'A1-LF_s{sigma:g}'/'flow.pt'
        fit_map(f'A1-LF_s{sigma:g}','A1-LF',sigma,'lf')
        fit_map(f'A1-GS_insample_s{sigma:g}','A1-GS',sigma,init=initial,small=True)
        fit_map(f'A2_s{sigma:g}','A2',sigma)
        fit_map(f'A2_euclidean_s{sigma:g}','A2',sigma,a2_cost='euclidean')
        fit_map(f'A2-EM_s{sigma:g}','A2-EM',sigma,a2_cost='euclidean')
    fit_map('direct_pre','direct',map_source='lf')
    direct_pre=root/'direct_pre/flow.pt'
    fit_map('direct_insample','direct',init=direct_pre,small=True)
    fit_map('direct','direct',map_source='split1',init=direct_pre,small=True)
    fit_map('A1-GS_oos_s0','A1-GS',map_source='split1',init=root/'A1-LF_s0/flow.pt',small=True)
    for c in candidates:
        deps=[Path(c.space_dir)/'lf_spaces.pt',Path(c.gs_dir)/'gs_encoder.pt']
        if c.arm!='A3': deps.append(Path(c.out)/'flow.pt')
        v=config(c,stage='validation',validation_only=True)
        run_stage(v,d,dev,inputs=deps,outputs=[Path(c.out)/'validation.json',Path(c.out)/'validation_gs_preds.npz'])
    selection=freeze_selection(a,candidates)
    baseline=config(a,arm='A0',out=str(root/'A0'),stage='eval')
    for c in [baseline]+candidates:
        c=config(c,stage='eval',validation_only=False)
        deps=[root/'selection.json']
        if c.arm=='A0': deps += [Path(a.data)/f'runs/n029_tt6c_x_{a.holdout}/gs_bind.pt']
        else:
            deps += [Path(c.space_dir)/'lf_spaces.pt',Path(c.gs_dir)/'gs_encoder.pt']
            if c.arm!='A3': deps.append(Path(c.out)/'flow.pt')
        run_stage(c,d,dev,inputs=deps,outputs=[Path(c.out)/'eval.json',Path(c.out)/'eval_gs_preds.npz'])
        selected=c.arm=='A0' or (c.arm in selection['selected'] and selection['selected'][c.arm]['run']==str(Path(c.out).resolve()))
        if selected and not a.skip_clinical:
            cc=config(c,stage='clinical')
            run_stage(cc,d,dev,inputs=deps+[Path(__file__).with_name('n054_clinical.py'),Path(__file__).with_name('n054_contrasts.py')],
                outputs=[Path(c.out)/'clinical/clinical.json',Path(c.out)/'clinical/auroc.csv'])
    comparisons=[]
    for arm,pre,post,insample in [('A1-GS','A1-LF_s0','A1-GS_oos_s0','A1-GS_insample_s0'),
                                 ('direct','direct_pre','direct','direct_insample')]:
        for phase,name in [('pre',pre),('post_oos',post),('post_insample_secondary',insample)]:
            ev=json.loads((root/name/'eval.json').read_text())
            comparisons.append(dict(arm=arm,phase=phase,run=str(root/name),M1_autofl_truth_r=ev['M1_autofl_truth_r'],
                R_at_1=ev['M3']['R_at_1'],mapped_R_at_1=ev['M3_mapped_GS_LFspace1']['R_at_1'],mapped_GS_LF_CKA=ev['M4_mapped_GS_LFspace1']['linear_cka']))
    json_write(root/'pre_post.json',dict(comparisons=comparisons,selection='fixed primary sigma0, final iterate; no split1 selection'))
    json_write(root/'unit.done.json',dict(code_signature=CODE_SIGNATURE,unit=a.unit,selection=artifact(root/'selection.json')))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('legacy_stage',nargs='?',choices=['all','lf_spaces','gs_encoder','gs_cache','flow','eval','completion'])
    p.add_argument('--stage',choices=['all','lf_spaces','gs_encoder','gs_cache','flow','eval','completion'],default='all')
    p.add_argument('--unit',choices=['lf','batch1','batch0'],default='lf')
    p.add_argument('--out',required=True); p.add_argument('--lf-unit',default=ROOT+'/runs/n054v3_lf')
    p.add_argument('--arm',choices=['A0','A1-LF','A1-GS','A2','A2-EM','A3','direct'],default='A3')
    p.add_argument('--space-dir',default=''); p.add_argument('--gs-dir',default=''); p.add_argument('--init-flow',default=''); p.add_argument('--pair-cache',default='')
    p.add_argument('--holdout',choices=['batch0','batch1'],default='batch1'); p.add_argument('--data',default=ROOT)
    p.add_argument('--targets',default='targets_tt6c.npz')
    p.add_argument('--lf-marker',default='runs/n029_r18_v4c_fov/probe'); p.add_argument('--lf-autofl',default='runs/n029_r18_v6c_fov/probe')
    p.add_argument('--map-source',choices=['lf','split0','split1'],default='split0')
    p.add_argument('--a2-cost',choices=['gw','euclidean'],default='gw')
    p.add_argument('--gs-map-steps',type=int,default=600)
    p.add_argument('--steps',type=int,default=2500); p.add_argument('--bs',type=int,default=128)
    p.add_argument('--lf-bs',type=int,default=256); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lf-lr',type=float,default=1e-3); p.add_argument('--flow-lr',type=float,default=1e-3)
    p.add_argument('--dim',type=int,default=256); p.add_argument('--sigma-bridge',type=float,choices=[0,.1],default=0)
    p.add_argument('--w-sig',type=float,choices=[0,.5],default=0); p.add_argument('--euler-steps',type=int,default=32)
    p.add_argument('--tau',type=float,default=.25); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda']); p.add_argument('--threads',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=250); p.add_argument('--smoke',action='store_true')
    p.add_argument('--ridge-cache',default=''); p.add_argument('--baseline',default=''); p.add_argument('--validation-only',action='store_true')
    p.add_argument('--skip-clinical',action='store_true',help='development only; production unit runs clinical on A0 and selected configs')
    return p


def main():
    a=parser().parse_args()
    if a.legacy_stage: a.stage=a.legacy_stage
    if a.arm=='A0' and a.stage not in ('all','eval'): raise ValueError('A0 is evaluation only')
    if a.smoke:
        a.steps=min(a.steps,20); a.bs=min(a.bs,4); a.lf_bs=min(a.lf_bs,16)
    torch.set_num_threads(a.threads); seed_all(a.seed)
    signal.signal(signal.SIGTERM,stop_handler); signal.signal(signal.SIGUSR1,stop_handler)
    dev=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    Path(a.out).mkdir(parents=True,exist_ok=True)
    if a.stage=='all': driver(a,dev)
    else:
        if a.stage=='completion': a.targets='targets_v5cx_v10.npz'
        d=Data(a); run_stage(a,d,dev)


if __name__=='__main__': main()
