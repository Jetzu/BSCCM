#!/usr/bin/env python3
"""N045: frozen-feature semi-supervised OT alignment; writes only below --out.

No image/fluorescence readers. SOTAlign's MatchingModel, CCA and AlignmentLayer
are loaded from the read-only upstream checkout. See README.md for protocols.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn
import torch.nn.functional as F
from lion_pytorch import Lion

ARMS = ('raw_ridge', 'procrustes', 'cca', 'siglip_pairs', 'sotalign',
        'sotalign_lf_only', 'n009_siglip')
PROTOCOLS = ('cross_b0to1', 'cross_b1to0', 'fov_b0', 'fov_b1')
# Fixed before any N045 outputs. Names/directions match N021/N032; two additions
# match the user's requested mono>rest and lym>neut comparisons.
CONTRASTS = {
    'CD16 neutrophil>lymphocyte': ('single:CD16', ['neutrophil'], ['lymphocyte']),
    'CD16 neutrophil>eosinophil': ('single:CD16', ['neutrophil'], ['eosinophil']),
    'CD14 monocyte>neutrophil+lymphocyte': ('single:CD14', ['monocyte'], ['neutrophil', 'lymphocyte']),
    'grpCD3/19/56 lymphocyte>neutrophil+monocyte': ('grp:CD3/CD19/CD56', ['lymphocyte'], ['neutrophil', 'monocyte']),
    'CD3 lymphocyte>neutrophil+monocyte': ('single:CD3', ['lymphocyte'], ['neutrophil', 'monocyte']),
    'CD45 lymphocyte>neutrophil': ('single:CD45', ['lymphocyte'], ['neutrophil']),
    'CD123 basophil>rest': ('single:CD123', ['basophil'], ['neutrophil', 'lymphocyte', 'monocyte', 'eosinophil']),
    'HLA-DR monocyte>neutrophil': ('single:HLA-DR', ['monocyte'], ['neutrophil']),
    'CD16 neutrophil>ig (label-held-out)': ('single:CD16', ['neutrophil'], ['ig']),
    'CD45 lymphocyte>erythroblast (label-held-out)': ('single:CD45', ['lymphocyte'], ['erythroblast']),
    'CD14 monocyte>all_other_classes': ('single:CD14', ['monocyte'], None),
    'CD3 lymphocyte>neutrophil': ('single:CD3', ['lymphocyte'], ['neutrophil']),
}
STOP = False


def stop_handler(signum, frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, torch.Tensor):
        return jsonable(x.detach().cpu().tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_torch(path, obj):
    tmp = Path(str(path) + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_csv(path, rows):
    keys = sorted({k for r in rows for k in r})
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(jsonable(rows))
    os.replace(tmp, path)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-10 or np.std(y[m]) < 1e-10:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def sample(rows, n, seed):
    rows = np.asarray(rows, np.int64)
    return np.sort(np.random.default_rng(seed).permutation(rows)[:n]) if n else np.sort(rows)


def upstream(root):
    """Execute the original class ASTs, avoiding unused heavyweight encoder imports.

    AlignmentLayer and L2Norm are compiled unchanged from sail_model.py. There
    are no dummy modules or replacement losses. linear.py is loaded unchanged.
    """
    src = root / 'src'
    sys.path.insert(0, str(src))
    from optimal_transport.matching import MatchingModel
    from optimal_transport.similarity import CCA, Anchors, Procrustes
    spec = importlib.util.spec_from_file_location('n045_upstream_linear', src / 'models/linear.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tree = ast.parse((src / 'models/sail_model.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('L2Norm', 'AlignmentLayer')]
    require(len(nodes) == 2, 'upstream AlignmentLayer layout changed')
    from functools import partial
    from typing import Optional
    scope = dict(torch=torch, nn=nn, F=F, os=os, CCA=CCA, Anchors=Anchors,
                 Procrustes=Procrustes, StarMLP=mod.StarMLP, SiglipMLP=mod.SiglipMLP,
                 partial=partial, Optional=Optional, __name__=__name__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src / 'models/sail_model.py'), 'exec'), scope)
    # Register classes so torch serialization of state_dicts needs no upstream package.
    globals()['AlignmentLayer'] = scope['AlignmentLayer']
    return MatchingModel, CCA, scope['AlignmentLayer']


def source_manifest(a):
    paths = [a.targets, a.giemsa, a.pbc, a.data_root / 'emb/pbc_labels.npy',
             a.data_root / 'emb/pbc_classes.json', a.data_root / 'pack/hist_cells.npy',
             a.lf_dir / 'lf_feats_all.npy', a.lf_dir / 'lf_feats_hist.npy',
             a.lf_dir / 'lf_feats_hist_indices.npy', a.lf_dir / 'lf_feats_hist_hist_row.npy']
    if a.legacy_truth.exists():
        paths.append(a.legacy_truth)
    # Hash metadata, targets and source; very large embedding files use stat identity.
    meta = {}
    for p in paths:
        st = p.stat()
        meta[str(p)] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        if st.st_size < 2_000_000 or p == a.targets:
            meta[str(p)]['sha256'] = digest(p)
    source_files = ['src/optimal_transport/matching.py', 'src/optimal_transport/similarity.py',
                    'src/optimal_transport/klot.py', 'src/optimal_transport/utils.py',
                    'src/models/sail_model.py', 'src/models/linear.py', 'scripts/sotalign.sh']
    return dict(inputs=meta, upstream_sha256={f: digest(a.sotalign_root / f) for f in source_files},
                runner_sha256=digest(__file__))


class Data:
    def __init__(self, a):
        self.a = a
        with np.load(a.targets, allow_pickle=False) as t:
            self.t = {k: t[k] for k in ('Y', 'mask', 'names', 'batch', 'slide', 'slide_names',
                                       'fov', 'split', 'hist_row', 'gs_pair_suspect')}
        t = self.t
        self.names = [str(n) for n in t['names']]
        self.h = np.load(a.data_root / 'pack/hist_cells.npy', allow_pickle=False)
        self.G = np.load(a.giemsa, mmap_mode='r', allow_pickle=False)
        self.L = np.load(a.lf_dir / 'lf_feats_all.npy', mmap_mode='r', allow_pickle=False)
        self.P = np.load(a.pbc, mmap_mode='r', allow_pickle=False)
        require(len(self.L) == len(t['Y']), 'LF rows must equal targets global indices')
        require(len(self.G) == len(self.h), 'GS rows must equal hist_cells length')
        require(np.array_equal(t['hist_row'][self.h], np.arange(len(self.h))), 'hist_cells / hist_row mismatch')
        require(np.array_equal(np.sort(self.h), np.flatnonzero(t['hist_row'] >= 0)), 'missing/duplicate GS cells')
        require(self.G.shape[1] == self.P.shape[1], 'PBC and BSCCM GS embedding dimensions differ')
        require(not t['mask'][self.h][:, [j for j, n in enumerate(self.names) if n != 'autofl']].any(),
                'K001 violated: a GS cell has a surface-marker truth label')
        ids = np.load(a.lf_dir / 'lf_feats_hist_indices.npy', allow_pickle=False)
        hr = np.load(a.lf_dir / 'lf_feats_hist_hist_row.npy', allow_pickle=False)
        lh = np.load(a.lf_dir / 'lf_feats_hist.npy', mmap_mode='r', allow_pickle=False)
        require(np.array_equal(t['hist_row'][ids], hr), 'LF hist indices/hist_row mismatch')
        require(np.array_equal(np.sort(ids), np.sort(self.h)), 'LF hist index coverage mismatch')
        require(len(lh) == len(ids), 'LF hist feature length mismatch')
        # Two independent extraction passes differ slightly; use global LF array
        # consistently for pairs, unpaired data and marker heads. Never combine passes.
        err, sq, count, mx = 0.0, 0.0, 0, 0.0
        for i in range(0, len(ids), 1024):
            x = np.asarray(lh[i:i+1024], np.float32)
            y = np.asarray(self.L[ids[i:i+1024]], np.float32)
            delta = np.abs(x - y)
            err += float(delta.sum()); sq += float((delta**2).sum()); count += delta.size
            mx = max(mx, float(delta.max()))
        require(mx < 0.05, f'LF extraction passes disagree by {mx}; investigate feature identity')
        self.universe = np.arange(len(self.h))
        if a.smoke:
            # Bounded, label-blind subset preserves each pre-existing FOV split.
            blocks = [sample(np.flatnonzero((t['batch'][self.h] == b) & (t['split'][self.h] == s)),
                             128, 90 + 10*int(b) + int(s)) for b in (0, 1) for s in (0, 1, 2)]
            self.universe = np.sort(np.concatenate(blocks))  # <=768 unique GS cells
        # PBC rows follow sorted archive filenames. A/B = alternating rows, as N021.
        all_p = np.arange(len(self.P))
        self.pa = sample(all_p[::2], a.max_pbc, 72)
        self.pb = sample(all_p[1::2], a.max_pbc, 73)
        require(not np.intersect1d(self.pa, self.pb).size, 'PBC adaptation/evaluation overlap')
        # Labels intentionally not loaded until evaluate(). Verify embedding row
        # identity using label-free filename count/order construction provenance.
        archive = a.data_root / 'external/pbc/PBC_dataset_normal_DIB.zip'
        self.pbc_files = None
        if archive.exists():
            with zipfile.ZipFile(archive) as z:
                self.pbc_files = sorted(n for n in z.namelist() if n.lower().endswith('.jpg')
                    and '__MACOSX' not in n and not Path(n).name.startswith('.'))
            require(len(self.pbc_files) == len(self.P), 'PBC archive/embedding row count mismatch')
        self.audit = dict(n_global=len(self.L), n_pairs=len(self.h),
            hist_mapping_exact=True, lf_hist_index_mapping_exact=True,
            lf_independent_pass_max_abs=mx, lf_independent_pass_mean_abs=err/count,
            lf_independent_pass_rmse=math.sqrt(sq/count), canonical_lf='lf_feats_all.npy',
            smoke_unique_gs_cells=len(self.universe), pbc_a_n=len(self.pa), pbc_b_n=len(self.pb),
            pbc_split='sorted original filename order, even adaptation A / odd evaluation B',
            giemsa_slides=[dict(draw=int(b), slide=int(s), name=str(t['slide_names'][s]),
                n=int(((t['batch'][self.h] == b) & (t['slide'][self.h] == s)).sum()))
                for b in np.unique(t['batch'][self.h])
                for s in np.unique(t['slide'][self.h][t['batch'][self.h] == b])],
            gs_pair_suspect_n=int(t['gs_pair_suspect'][self.h].sum()),
            test_suspects_filtered=False)
        # Frozen LF teacher provenance; fail closed when a known supervised run
        # included Giemsa cells. SSL alternatives have no train_lf result.json.
        prov = a.lf_dir.parent / 'result.json'
        if prov.exists():
            args = json.loads(prov.read_text()).get('args', {})
            require(args.get('include_gs') == 0, 'LF teacher provenance must explicitly exclude GS cells')
            self.audit['lf_teacher_args'] = args
        else:
            require(a.allow_unverified_lf_provenance, 'missing LF teacher provenance; supply verified features or explicit --allow-unverified-lf-provenance')
            self.audit['lf_teacher_provenance'] = 'user-attested; not machine verified'
        # Check FOVs really are globally split before trusting row-level masks.
        fsort = np.argsort(t['fov'], kind='stable')
        same = t['fov'][fsort[1:]] == t['fov'][fsort[:-1]]
        require(np.all(t['split'][fsort[1:]][same] == t['split'][fsort[:-1]][same]), 'one FOV has multiple split labels')

    def partition(self, protocol, seed):
        t, h = self.t, self.h
        b = 0 if protocol in ('cross_b0to1', 'fov_b0') else 1
        u = self.universe
        source = t['batch'][h[u]] == b
        tr = sample(u[source & (t['split'][h[u]] == 0)], self.a.max_pairs, seed)
        va = sample(u[source & (t['split'][h[u]] == 1)], self.a.max_val_pairs, 170)
        if protocol.startswith('cross'):
            te = u[t['batch'][h[u]] != b]
        else:
            te = u[source & (t['split'][h[u]] == 2)]
        eligible = (t['hist_row'] < 0) & (t['batch'] == b)
        uy = sample(np.flatnonzero(eligible & (t['split'] == 0)), self.a.n_unpaired, seed+71)
        head_tr = sample(np.flatnonzero(eligible & (t['split'] == 0) & t['mask'].any(1)), self.a.head_train, seed+81)
        head_va = sample(np.flatnonzero(eligible & (t['split'] == 1) & t['mask'].any(1)), self.a.head_val, 181)
        require(min(len(tr), len(va), len(te), len(uy)) >= 3, 'too few cells for a partition')
        require(not np.intersect1d(tr, va).size and not np.intersect1d(tr, te).size, 'paired partition overlap')
        require(not np.intersect1d(t['fov'][h[tr]], t['fov'][h[va]]).size, 'train/validation FOV overlap')
        require(not np.intersect1d(t['fov'][h[tr]], t['fov'][h[te]]).size, 'train/test FOV overlap')
        for rows in (uy, head_tr, head_va):
            require(np.all(t['hist_row'][rows] < 0) and np.all(t['batch'][rows] == b), 'LF pool is GS or from test draw')
        return dict(train=tr, val=va, test=te, unpaired_y=uy, head_train=head_tr, head_val=head_va,
                    source_draw=b, pbc_a=self.pa, pbc_b=self.pb)

    def X(self, rows):
        return np.asarray(self.G[rows], np.float32)

    def Y(self, rows):
        return np.asarray(self.L[self.h[rows]], np.float32)


class FrozenMap:
    def __init__(self, kind, state):
        self.kind, self.state = kind, state

    def encode(self, x, side):
        s = self.state
        if self.kind == 'cca':
            c = s['cca']
            with torch.no_grad():
                z = (F.normalize(torch.from_numpy(np.asarray(x, np.float32)), dim=1)
                     - getattr(c, side+'_mean').cpu()) @ getattr(c, 'W'+side).cpu()
            return norm(z.numpy())
        if self.kind == 'raw_ridge':
            return norm(s['ridge'].predict(s['scale'].transform(x))) if side == 'x' else norm(x)
        z = s['pca_'+side].transform(x)
        if side == 'x':
            z = z @ s['rotation']
        return norm(z)


class LearnedMap:
    def __init__(self, model, a, px=None, py=None):
        self.model, self.a, self.px, self.py = model, a, px, py

    def prepare(self, x, side):
        p = self.px if side == 'x' else self.py
        return p.transform(x).astype(np.float32) if p is not None else np.asarray(x, np.float32)

    def encode(self, x, side):
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 2048):
                z = torch.as_tensor(self.prepare(x[i:i+2048], side), device=self.a.device)
                d = self.model(**{'text_features' if side == 'x' else 'image_features': z})
                out.append(F.normalize(d['text_features' if side == 'x' else 'image_features'], dim=1).cpu().numpy())
        result = np.concatenate(out)
        require(np.isfinite(result).all(), 'nonfinite embedding')
        return result


def retrieval(x, y, device='cpu', chunk=256):
    require(len(x) == len(y) and len(x) > 0, 'retrieval needs paired, nonempty arrays')
    x = torch.as_tensor(norm(x), device=device)
    y = torch.as_tensor(norm(y), device=device)
    hits1, hits10, ranks = 0, 0, []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            sim = x[i:i+chunk] @ y.T
            diag = (x[i:i+chunk] * y[i:i+chunk]).sum(1)
            # Pessimistic tie handling avoids optimistic retrieval for collapse.
            rank = (sim >= diag[:, None] - 1e-6).sum(1).clamp_min(1)
            hits1 += int((rank <= 1).sum()); hits10 += int((rank <= 10).sum())
            ranks.extend(rank.cpu().tolist())
    return dict(R1=hits1/len(x), R10=hits10/len(x), median_rank=float(np.median(ranks)),
                n_test=len(x), chance_R1=1/len(x), chance_R10=min(10/len(x), 1.0))


def fit_pcas(d, part, a):
    n = min(a.dim, len(part['train'])-1, d.G.shape[1], d.L.shape[1])
    px = PCA(n_components=n, whiten=True, svd_solver='full').fit(d.X(part['train']))
    py = PCA(n_components=n, whiten=True, svd_solver='full').fit(d.Y(part['train']))
    return px, py


def fit_cca(d, part, a, CCA):
    c = CCA(n_components=min(a.dim, len(part['train'])-1, d.G.shape[1], d.L.shape[1]),
            lam_x=a.cca_lam, lam_y=a.cca_lam, eps=1e-6)
    with torch.no_grad():
        c.fit(torch.as_tensor(d.X(part['train']), device=a.device),
              torch.as_tensor(d.Y(part['train']), device=a.device))
    return c


def train_model(d, part, a, arm, seed, folder, cca, MatchingModel, AlignmentLayer, pcas):
    n009 = arm == 'n009_siglip'
    px, py = pcas if n009 else (None, None)
    torch.manual_seed(seed)
    dx = px.n_components_ if n009 else d.G.shape[1]
    dy = py.n_components_ if n009 else d.L.shape[1]
    model = AlignmentLayer(vision_dimension=dy, text_dimension=dx,
        target_dimension=min(a.dim, len(part['train'])-1, dy, dx), linear_type='linear',
        cast_dtype=torch.float32, logit_scale=20., logit_bias=-10., width_factor=1.).to(a.device)
    if n009:
        model.text_layer_norm = nn.Identity()
        model.vision_layer_norm = nn.Identity()
    mapping = LearnedMap(model, a, px, py)
    sot = arm in ('sotalign', 'sotalign_lf_only')
    expanded = sot and a.sail_negatives == 'expanded'
    cfg = dict(alpha_supervised_sail=0. if expanded else 1., alpha_supervised_implicit=0.,
        alpha_semisupervised_sail=1. if expanded else 0., alpha_semisupervised_klot=a.klot_weight if sot else 0.,
        alpha_semisupervised_softmax=0., alpha_semisupervised_cka=0.,
        epsilon_sinkhorn_student=a.eps_student, epsilon_sinkhorn_teacher=a.eps_teacher,
        n_iters_sinkhorn_student=a.sinkhorn_iters, n_iters_sinkhorn_teacher=a.sinkhorn_iters,
        use_stop_criterion=False)
    matching = MatchingModel(cfg).to(a.device)
    matching.affinity = cca
    optimizer = (Lion(model.parameters(), lr=a.lr, betas=(.9, .99), weight_decay=a.wd)
                 if a.optimizer == 'lion' else torch.optim.AdamW(model.parameters(), lr=a.lr,
                    betas=(.9, .99), weight_decay=a.wd))
    x = torch.as_tensor(mapping.prepare(d.X(part['train']), 'x'), device=a.device)
    y = torch.as_tensor(mapping.prepare(d.Y(part['train']), 'y'), device=a.device)
    ux = torch.as_tensor(np.asarray(d.P[part['pbc_a']], np.float32), device=a.device) if arm == 'sotalign' else x
    uy = torch.as_tensor(np.asarray(d.L[part['unpaired_y']], np.float32), device=a.device) if sot else None
    last, best_path = folder / 'last.pt', folder / 'best.pt'
    start, best, best_step, history = 0, -1., 0, []
    if a.resume and last.exists():
        ck = torch.load(last, map_location=a.device, weights_only=False)
        require(ck['fingerprint'] == a.fingerprint, 'checkpoint configuration mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        start, best, best_step, history = ck['step'], ck['best'], ck['best_step'], ck['history']
    def save(step):
        atomic_torch(last, dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
            best=best, best_step=best_step, history=history, fingerprint=a.fingerprint))
    begin = time.time()
    for step in range(start, a.steps):
        model.train()
        # Separate per-step streams guarantee identical pair minibatches across
        # all trainable arms and exact continuation without saving sampler state.
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 11]))
        idx = rng.choice(len(x), min(a.batch_pairs, len(x)), replace=False)
        xp, yp = x[idx], y[idx]
        fp = model(text_features=xp, image_features=yp)
        uxx = uyy = fux = fuy = None
        if sot:
            rux = np.random.default_rng(np.random.SeedSequence([seed, step, 12]))
            ruy = np.random.default_rng(np.random.SeedSequence([seed, step, 13]))
            # LF-only: X has no additional unpaired data. Training GS anchors
            # supply the X support for the rectangular OT against extra LF cells.
            uxx = ux[rux.choice(len(ux), min(a.batch_unpaired, len(ux)), replace=False)]
            uyy = uy[ruy.choice(len(uy), min(a.batch_unpaired, len(uy)), replace=False)]
            fu = model(text_features=uxx, image_features=uyy)
            fux, fuy = fu['text_features'], fu['image_features']
            if expanded and arm == 'sotalign_lf_only':
                # Do not label reused X anchors as extra negative examples.
                loss_sail = matching.loss_semisupervised_sail(fp['text_features'], fp['image_features'],
                    None, fuy, fp['logit_scale'], fp['logit_bias'])
                ks = F.normalize(fux, dim=1) @ F.normalize(fuy, dim=1).T
                kt = cca(uxx, uyy)
                kl, es, et = matching.loss_semisupervised_klot(ks, kt)
                loss = loss_sail + a.klot_weight * kl
                log = dict(total_loss=loss.item(), loss_semisupervised_sail=loss_sail.item(),
                    loss_semisupervised_klot=kl.item(), sinkhorn_err_student=es, sinkhorn_err_teacher=et)
            else:
                loss, log = matching(uxx, uyy, xp, yp, fux, fuy,
                                    fp['text_features'], fp['image_features'], fp['logit_scale'], fp['logit_bias'])
        else:
            loss, log = matching(None, None, xp, yp, None, None,
                                fp['text_features'], fp['image_features'], fp['logit_scale'], fp['logit_bias'])
        require(bool(torch.isfinite(loss)), f'nonfinite training loss {arm} step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(), 2.)
        require(bool(torch.isfinite(grad)), 'nonfinite gradient')
        # Upstream cosine schedule, same prescribed budget for every arm.
        for group in optimizer.param_groups:
            group['lr'] = a.lr * .5 * (1 + math.cos(math.pi * step / max(a.steps, 1)))
        optimizer.step()
        done = step+1
        if done % a.eval_every == 0 or done == a.steps or STOP:
            val = retrieval(mapping.encode(d.X(part['val']), 'x'), mapping.encode(d.Y(part['val']), 'y'), a.device)
            r = dict(step=done, seconds_this_invocation=time.time()-begin, val_R10=val['R10'],
                     gradient_norm=float(grad), lr=optimizer.param_groups[0]['lr'], **log)
            if sot:
                r['sinkhorn_teacher_relative_row_error'] = float(r.get('sinkhorn_err_teacher', 0)) * len(uxx)
                r['sinkhorn_student_relative_row_error'] = float(r.get('sinkhorn_err_student', 0)) * len(uxx)
            history.append(jsonable(r))
            if val['R10'] > best:
                best, best_step = val['R10'], done
                atomic_torch(best_path, dict(model=model.state_dict(), step=done, val=val,
                                             fingerprint=a.fingerprint))
            save(done)
            atomic_json(folder / 'training.json', dict(config=cfg, history=history, best_step=best_step, best_val_R10=best))
            print(json.dumps(jsonable(dict(arm=arm, seed=seed, **r))), flush=True)
        elif done % a.checkpoint_every == 0:
            save(done)
        if STOP:
            save(done)
            raise SystemExit(143)   # red-team M1: preempt-run resubmits only on 0 or 143 after a signal
    require(best_path.exists(), 'no best checkpoint; need >=1 training step')
    ck = torch.load(best_path, map_location=a.device, weights_only=False)
    require(ck['fingerprint'] == a.fingerprint, 'best checkpoint mismatch')
    model.load_state_dict(ck['model'])
    return mapping, dict(best_step=best_step, best_val_R10=best, loss_config=cfg)


def marker_heads(d, part, a, mapping, folder):
    """Labels enter only here, after alignment fitting. Each target is masked."""
    t = d.t
    tr, va = part['head_train'], part['head_val']
    xt = mapping.encode(np.asarray(d.L[tr], np.float32), 'y')
    xv = mapping.encode(np.asarray(d.L[va], np.float32), 'y')
    scaler = StandardScaler().fit(xt)
    xt, xv = scaler.transform(xt), scaler.transform(xv)
    heads, meta = [], []
    for j, name in enumerate(d.names):
        mt = t['mask'][tr, j] & np.isfinite(t['Y'][tr, j])
        mv = t['mask'][va, j] & np.isfinite(t['Y'][va, j])
        info = dict(target=name, n_train=int(mt.sum()), n_val=int(mv.sum()),
                    status='ok', alpha=None, val_mse=None)
        if mt.sum() < a.min_head_labels or mv.sum() < 3 or np.std(t['Y'][tr[mt], j]) < 1e-8:
            heads.append(None); info['status'] = 'insufficient_train_or_validation_labels'
        else:
            best_mse, head = float('inf'), None
            for alpha in a.ridge_alphas:
                candidate = Ridge(alpha=alpha).fit(xt[mt], t['Y'][tr[mt], j])
                mse = float(np.mean((candidate.predict(xv[mv]) - t['Y'][va[mv], j])**2))
                if mse < best_mse:
                    head, best_mse = candidate, mse
                    info.update(alpha=alpha, val_mse=mse)
            heads.append(head)
        meta.append(info)
    atomic_json(folder / 'marker_heads.json', meta)
    np.savez(folder / 'marker_heads.npz', mean=scaler.mean_, scale=scaler.scale_,
        coef=np.stack([h.coef_ if h is not None else np.full(xt.shape[1], np.nan) for h in heads]),
        intercept=np.array([float(h.intercept_) if h is not None else np.nan for h in heads]),
        names=np.array(d.names))
    def predict(z):
        zz = scaler.transform(z)
        return np.stack([h.predict(zz) if h is not None else np.full(len(z), np.nan) for h in heads], axis=1)
    return predict, meta


def pbc_labels(d):
    # Evaluation-only access. Neither labels nor classes are returned to trainers.
    classes = json.loads((d.a.data_root / 'emb/pbc_classes.json').read_text())
    ids = np.load(d.a.data_root / 'emb/pbc_labels.npy', allow_pickle=False)
    require(ids.shape == (len(d.P),) and ids.min() >= 0 and ids.max() < len(classes), 'bad PBC labels')
    if d.pbc_files is not None:
        check = np.array([classes.index(n.split('/')[-2]) for n in d.pbc_files])
        require(np.array_equal(ids, check), 'PBC labels disagree with sorted archive rows')
    return np.array(classes)[ids]


def evaluate(d, part, a, mapping, folder, base):
    tr, va, te = part['train'], part['val'], part['test']
    zx, zy = mapping.encode(d.X(te), 'x'), mapping.encode(d.Y(te), 'y')
    val = retrieval(mapping.encode(d.X(va), 'x'), mapping.encode(d.Y(va), 'y'), a.device)
    retr = retrieval(zx, zy, a.device)
    predict, head_meta = marker_heads(d, part, a, mapping, folder)
    pg, pl = predict(zx), predict(zy)
    rows = [dict(base, family='retrieval', metric=k, value=v, n=len(te), split='test')
            for k, v in retr.items() if k not in ('n_test',)]
    rows.append(dict(base, family='retrieval', metric='R10', value=val['R10'], n=len(va), split='source_validation'))
    t = d.t; global_te = d.h[te]; ja = d.names.index('autofl')
    primary_truth = t['Y'][global_te, ja]
    valid = t['mask'][global_te, ja]
    rows.extend([dict(base, family='autofl_truth', metric=metric,
        value=pearson(p[valid, ja], primary_truth[valid]), n=int(valid.sum()),
        truth='targets_v5c_position_detrended_autofl', split='test')
        for metric, p in [('GS_head_transfer_pearson_r', pg), ('same_cell_LF_head_pearson_r', pl)]])
    # Legacy N012 truth is evaluation-only and never used for any fit/choice.
    if a.legacy_truth.exists():
        legacy = np.load(a.legacy_truth, allow_pickle=False)
        require(legacy.shape == (len(d.h),), 'N012 truth must be in hist_row order')
        for metric, p in [('GS_head_transfer_pearson_r', pg), ('same_cell_LF_head_pearson_r', pl)]:
            rows.append(dict(base, family='autofl_truth', metric=metric, value=pearson(p[:, ja], legacy[te]),
                n=len(te), truth='N012_region_crossfitted_residual_legacy_preclean', split='test'))
    for j, name in enumerate(d.names):
        rows.append(dict(base, family='teacher_agreement_NOT_marker_truth', metric='GS_LF_prediction_pearson_r',
                         target=name, value=pearson(pg[:, j], pl[:, j]), n=len(te), split='test'))
    pa, pb = part['pbc_a'], part['pbc_b']
    # Transform A here even for controls: labels are used for fixed-k sanity
    # evaluation only, after all model and ridge hyperparameters have frozen.
    za = mapping.encode(np.asarray(d.P[pa], np.float32), 'x')
    zb = mapping.encode(np.asarray(d.P[pb], np.float32), 'x')
    pp = predict(zb)
    labels = pbc_labels(d)
    knn = KNeighborsClassifier(n_neighbors=min(5, len(pa)), metric='cosine', algorithm='brute')
    kp = knn.fit(za, labels[pa]).predict(zb)
    for metric, value in [('knn_accuracy', accuracy_score(labels[pb], kp)),
                          ('knn_balanced_accuracy', balanced_accuracy_score(labels[pb], kp))]:
        rows.append(dict(base, family='PBC_sanity', metric=metric, value=float(value), n=len(pb), split='PBC_B'))
    for contrast, (marker, pos, neg) in CONTRASTS.items():
        j = d.names.index(marker)
        ispos = np.isin(labels[pb], pos)
        eligible = np.ones(len(pb), bool) if neg is None else np.isin(labels[pb], pos+neg)
        eligible &= np.isfinite(pp[:, j])
        value = float(roc_auc_score(ispos[eligible], pp[eligible, j])) if len(np.unique(ispos[eligible])) == 2 else None
        rows.append(dict(base, family='PBC_marker_contrast', metric='AUROC', target=marker,
            contrast=contrast, value=value, n=int(eligible.sum()), n_pos=int((eligible & ispos).sum()),
            n_neg=int((eligible & ~ispos).sum()), split='PBC_B',
            status='ok' if value is not None else 'missing_head_or_class'))
    np.savez_compressed(folder / 'predictions.npz', test_hist_row=te, test_global_index=global_te,
        gs_pred=pg, lf_pred=pl, target_names=np.array(d.names), truth_autofl=primary_truth,
        truth_autofl_mask=valid, pbc_b_rows=pb, pbc_pred=pp)
    # Geometry diagnostics, no marker truth implied.
    geometry = dict(gs_unit_norm_mean=float(np.linalg.norm(zx, axis=1).mean()),
        lf_unit_norm_mean=float(np.linalg.norm(zy, axis=1).mean()),
        gs_mean_feature_variance=float(zx.var(0).mean()), lf_mean_feature_variance=float(zy.var(0).mean()))
    atomic_json(folder / 'geometry.json', geometry)
    return rows


def raw_sanity(d, a):
    labels = pbc_labels(d)
    pa, pb = d.pa, d.pb
    model = KNeighborsClassifier(n_neighbors=min(5, len(pa)), metric='cosine', algorithm='brute')
    pred = model.fit(norm(d.P[pa]), labels[pa]).predict(norm(d.P[pb]))
    return [dict(arm='raw_giemsa', protocol='independent_PBC_sanity', seed=-1, family='PBC_sanity',
                 split='PBC_B', metric='knn_accuracy', value=float(accuracy_score(labels[pb], pred)), n=len(pb))]


def historical_references(a):
    rows = []
    res = a.repo / '.agent-research/results'
    screen = res / 'N009-stage-2-binding-v0-on-frozen-sup-lf-feat/screen_B/screen.json'
    if screen.exists():
        for r in json.loads(screen.read_text()):
            if r.get('n') == 10015 and r.get('protocol') == 'slide_b0_to_b1' and r.get('method') in ('siglip_linear', 'procrustes'):
                rows.append(dict(arm='historical_N009_'+r['method'], protocol='historical_cross_b0to1', seed=r['seed'],
                    family='retrieval', metric='R10', value=r['GS->LF_R@10'], n=r['n_test'],
                    source=str(screen), comparable=False, caveat='different LF teacher, pair count and preprocessing; context only'))
    for name, protocol, value, caveat in [
        ('K011_distilled_seed0', 'historical_cross_b0to1', .648, 'legacy detrended truth, different teacher'),
        ('K011_distilled_seed1', 'historical_cross_b0to1', .658, 'legacy detrended truth, different teacher'),
        ('K011_distilled', 'historical_cross_b1to0', .32, 'legacy detrended truth, different teacher'),
        ('K022_N029_TT6c', 'historical_fov_b0', .564, 'clean TT6c student with PBC class anchors; different training'),
        ('K022_N029_TT6c', 'historical_fov_b1', .848, 'clean TT6c student with PBC class anchors; different training')]:
        rows.append(dict(arm=name, protocol=protocol, seed=-1, family='autofl_truth',
            metric='GS_head_transfer_or_student_pearson_r', value=value, comparable=False, caveat=caveat,
            source=str(a.repo / '.agent-research/notebook/KNOWLEDGE.md')))
    for filename in ['N021-class-anchored-alignment-to-pbc-anchors/pbc_anchor_auroc.csv',
                     'N032-teacher-run-variance-of-clinical-transfe/transfer_all.csv']:
        path = res / filename
        if not path.exists():
            continue
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                if 'set' in r and r['set'] != 'PBC-B':
                    continue
                rows.append(dict(arm='historical_'+filename[:4]+'_'+r['arm'], protocol='historical_PBC_B',
                    seed=int(r['seed']), family='PBC_marker_contrast', metric='AUROC',
                    contrast=r['contrast'], value=float(r.get('value', r.get('auroc'))), source=str(path),
                    comparable=False, caveat='class-anchored image student; labels used for adaptation; context only'))
    return rows


def aggregate(out, refs):
    raw_file = out / 'raw_pbc_sanity.json'
    rows = json.loads(raw_file.read_text()) if raw_file.exists() else []
    for path in sorted(out.glob('*/seed*/**/metrics.json')):
        rows.extend(json.loads(path.read_text()))
    # Reference table includes contexts without mixing them into current means.
    write_csv(out / 'metrics.csv', rows)
    write_csv(out / 'comparison.csv', rows + refs)
    atomic_json(out / 'metrics.json', rows)
    groups = {}
    fields = ('protocol', 'arm', 'family', 'metric', 'target', 'contrast', 'truth', 'split')
    for r in rows:
        if r.get('value') is not None:
            groups.setdefault(tuple(r.get(k, '') for k in fields), []).append(float(r['value']))
    summary = [dict(zip(fields, key), n_seeds=len(vals), mean=float(np.mean(vals)),
        sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else None) for key, vals in groups.items()]
    write_csv(out / 'summary.csv', summary)


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    MatchingModel, CCA, AlignmentLayer = upstream(a.sotalign_root)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    if a.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
    manifest = source_manifest(a)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ('resume', 'fingerprint')}
    config['environment'] = dict(torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                                  sklearn=sklearn.__version__, device=a.device)
    identity = dict(config=config, sources=manifest)
    a.fingerprint = hashlib.sha256(json.dumps(jsonable(identity), sort_keys=True).encode()).hexdigest()
    existing = a.out / 'manifest.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        require(a.resume and old['fingerprint'] == a.fingerprint, 'output exists: use --resume with identical command/environment, or a new --out')
    atomic_json(existing, dict(identity, fingerprint=a.fingerprint))
    d = Data(a)
    atomic_json(a.out / 'mapping_checks.json', d.audit)
    atomic_json(a.out / 'preregistered_contrasts.json', CONTRASTS)
    refs = historical_references(a)
    atomic_json(a.out / 'historical_references.json', refs)
    consumed = set()
    for protocol in a.protocols:
        for seed in a.seeds:
            part = d.partition(protocol, seed)
            root = a.out / protocol / f'seed{seed}'
            root.mkdir(parents=True, exist_ok=True)
            np.savez(root / 'partitions.npz', **part)
            for k in ('train', 'val', 'test'):
                consumed.update(d.h[part[k]].tolist())
            for k in ('unpaired_y', 'head_train', 'head_val'):
                consumed.update(part[k].tolist())
            atomic_json(root / 'partition_audit.json', dict(source_draw=part['source_draw'],
                counts={k: len(v) for k, v in part.items() if isinstance(v, np.ndarray)},
                test_slide_excluded_from_all_training_pools=True,
                head_training_draw_only=True, no_GS_in_LF_unpaired_or_head=True))
            cca = None; pcas = None
            for arm in a.arms:
                folder = root / arm; folder.mkdir(exist_ok=True)
                if a.resume and (folder / 'metrics.json').exists():
                    continue
                if STOP:
                    raise SystemExit(143)   # red-team M1: preempt-run resubmits only on 0 or 143 after a signal
                if arm in ('cca', 'sotalign', 'sotalign_lf_only') and cca is None:
                    cca = fit_cca(d, part, a, CCA)
                    atomic_torch(root / 'cca_teacher.pt', {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in vars(cca).items()})
                meta = {}
                if arm == 'cca':
                    mapping = FrozenMap('cca', dict(cca=cca))
                elif arm == 'raw_ridge':
                    scaler = StandardScaler().fit(d.X(part['train']))
                    xt, y = scaler.transform(d.X(part['train'])), d.Y(part['train'])
                    best_r, best_head = -1., None
                    for alpha in a.ridge_alphas:
                        head = Ridge(alpha=alpha).fit(xt, y)
                        r = retrieval(head.predict(scaler.transform(d.X(part['val']))), d.Y(part['val']), a.device)['R10']
                        if r > best_r:
                            best_r, best_head = r, head
                            meta = dict(alpha=alpha, val_R10=r)
                    mapping = FrozenMap(arm, dict(scale=scaler, ridge=best_head))
                    import joblib
                    joblib.dump(mapping, folder / 'map.joblib')
                elif arm == 'procrustes':
                    if pcas is None:
                        pcas = fit_pcas(d, part, a)
                    px, py = pcas
                    xx, yy = px.transform(d.X(part['train'])), py.transform(d.Y(part['train']))
                    u, _, vt = np.linalg.svd(xx.T @ yy, full_matrices=False)
                    mapping = FrozenMap(arm, dict(pca_x=px, pca_y=py, rotation=u @ vt))
                    import joblib
                    joblib.dump(mapping, folder / 'map.joblib')
                else:
                    if arm == 'n009_siglip' and pcas is None:
                        pcas = fit_pcas(d, part, a)
                    mapping, meta = train_model(d, part, a, arm, seed, folder, cca,
                                               MatchingModel, AlignmentLayer, pcas)
                    if arm == 'n009_siglip':
                        import joblib
                        joblib.dump(pcas, folder / 'pcas.joblib')
                base = dict(protocol=protocol, seed=seed, arm=arm, smoke=a.smoke,
                            lf_teacher=str(a.lf_dir), train_pairs=len(part['train']))
                rows = evaluate(d, part, a, mapping, folder, base)
                atomic_json(folder / 'selection.json', meta)
                atomic_json(folder / 'metrics.json', rows)
                aggregate(a.out, refs)
                print(json.dumps(dict(event='arm_complete', **base,
                    retrieval_R10=next(r['value'] for r in rows if r['family']=='retrieval' and r['metric']=='R10' and r['split']=='test'))), flush=True)
                del mapping
                if a.device == 'cuda':
                    torch.cuda.empty_cache()
    if not (a.out / 'raw_pbc_sanity.json').exists():
        atomic_json(a.out / 'raw_pbc_sanity.json', raw_sanity(d, a))
    aggregate(a.out, refs)
    smoke_budget = dict(unique_bsccm_cells=len(consumed), unique_pbc_cells=len(d.pa)+len(d.pb),
                        total_unique_cells=len(consumed)+len(d.pa)+len(d.pb), unique_gs_cells=len(d.universe))
    if a.smoke:
        require(smoke_budget['total_unique_cells'] <= 2000, 'smoke exceeded 2000 unique cells')
    atomic_json(a.out / 'completion.json', dict(completed=True, smoke=a.smoke, sample_budget=smoke_budget,
        protocols=a.protocols, seeds=a.seeds, arms=a.arms, note='single supplied LF teacher; alignment seeds are not teacher replications'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/hnvme/workspace/v181bb10-bsccm'))
    p.add_argument('--repo', type=Path, default=Path('/home/hpc/v181bb/v181bb10/BSCCM'))
    p.add_argument('--sotalign-root', type=Path, default=Path('/hnvme/workspace/v181bb10-bsccm/external/SOTAlign'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lf-dir', type=Path)
    p.add_argument('--giemsa', type=Path)
    p.add_argument('--pbc', type=Path)
    p.add_argument('--targets', type=Path)
    p.add_argument('--legacy-truth', type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=list(PROTOCOLS))
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--batch-pairs', type=int, default=256)
    p.add_argument('--batch-unpaired', type=int, default=256)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--optimizer', choices=['lion', 'adamw'], default='lion')
    p.add_argument('--cca-lam', type=float, default=.1)
    p.add_argument('--klot-weight', type=float, default=.001)
    p.add_argument('--eps-student', type=float, default=.05)
    p.add_argument('--eps-teacher', type=float, default=.01)
    p.add_argument('--sinkhorn-iters', type=int, default=100)
    p.add_argument('--sail-negatives', choices=['expanded', 'pairs'], default='expanded')
    p.add_argument('--n-unpaired', type=int, default=100000)
    p.add_argument('--max-pairs', type=int, default=0)
    p.add_argument('--max-val-pairs', type=int, default=0)
    p.add_argument('--max-pbc', type=int, default=0, help='per half, label-blind cap; 0=all')
    p.add_argument('--head-train', type=int, default=100000)
    p.add_argument('--head-val', type=int, default=10000)
    p.add_argument('--min-head-labels', type=int, default=30)
    p.add_argument('--ridge-alphas', nargs='+', type=float, default=[1., 10., 100., 1000.])
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--allow-unverified-lf-provenance', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='fixed tiny real-data audit; <=2000 unique cells')
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.smoke:
        a.steps, a.batch_pairs, a.batch_unpaired, a.dim = 6, 32, 32, 32
        a.n_unpaired, a.head_train, a.head_val = 128, 128, 64
        a.max_pbc, a.min_head_labels, a.eval_every = 128, 3, 3
        a.max_pairs, a.max_val_pairs = 0, 0
        a.checkpoint_every = 3
        a.seeds = [0]
    a.lf_dir = a.lf_dir or a.data_root / 'runs/n029_r18_v4c_fov/probe'
    a.giemsa = a.giemsa or a.data_root / 'emb/giemsa_dinobloom_B.npy'
    a.pbc = a.pbc or a.data_root / 'emb/pbc_dinobloom_B.npy'
    a.targets = a.targets or a.data_root / 'targets_v5c.npz'
    a.legacy_truth = a.legacy_truth or a.data_root / 'gs_autofl_detrended.npy'
    for k in ('steps', 'batch_pairs', 'batch_unpaired', 'dim', 'sinkhorn_iters', 'eval_every', 'checkpoint_every', 'threads'):
        require(getattr(a, k) > 0, f'{k} must be positive')
    for k in ('max_pairs', 'max_val_pairs', 'max_pbc', 'n_unpaired', 'head_train', 'head_val'):
        require(getattr(a, k) >= 0, f'{k} must be nonnegative')
    require(all(x >= 0 for x in a.seeds), 'seeds must be nonnegative')
    require(a.lr > 0 and a.cca_lam > 0 and a.eps_student > 0 and a.eps_teacher > 0, 'invalid scale')
    require(a.klot_weight >= 0 and all(x > 0 for x in a.ridge_alphas), 'invalid loss/ridge weight')
    require(len(set(a.protocols)) == len(a.protocols) and len(set(a.arms)) == len(a.arms), 'duplicate protocol/arm')
    # Prevent accidental writes inside either read-only input tree, including via symlinks.
    target = a.out.resolve()
    for root in (a.repo.resolve(), a.data_root.resolve(), a.sotalign_root.resolve()):
        require(not target.is_relative_to(root), 'output must be outside all source/data trees')
    return a


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    args = parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)
