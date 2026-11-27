"""Load the registered teacher pair and the original N008 frozen image coordinates."""
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from scipy.special import softmax
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from n093_match import LF_CLASSES, sample


def targets(data):
    with np.load(data/'targets_v4c.npz') as a, np.load(data/'targets_v6c.npz') as b:
        t = {k: a[k] for k in ('names', 'hist_row', 'batch', 'slide', 'split', 'cls3_names')}
        for k in t:
            if not np.array_equal(t[k], b[k]):
                raise ValueError(f'target row or schema mismatch: {k}')
        if tuple(t['cls3_names']) != LF_CLASSES or len(t['names']) != 13 or t['names'][-1] != 'autofl':
            raise ValueError('unexpected BSCCM target schema')
        t['cls3'] = a['cls3']
        t['autofl'], t['observed'] = b['Y'][:, -1], b['mask'][:, -1]
    h = t['hist_row']
    if not np.array_equal(np.sort(h[h >= 0]), np.arange(15712)):
        raise ValueError('expected one global row per N054 paired cell')
    return t


def partition(t, holdout):
    h, b, s = t['hist_row'], t['batch'], t['split']
    test = np.flatnonzero((h >= 0) & (b == holdout))
    gs = np.flatnonzero((h >= 0) & (b == 1-holdout) & (s == 0))
    lf = np.flatnonzero((h < 0) & (b == 1-holdout) & (s == 0))
    slides = np.unique(t['slide'][test])
    if len(slides) != 1 or not len(gs) or not len(lf) or np.any(t['slide'][np.r_[gs, lf]] == slides[0]):
        raise ValueError('invalid held-out slide partition')
    if np.unique(t['slide'][gs]).size != 1:
        raise ValueError('paired router training rows span multiple GS slides')
    return gs, lf, test


def archived_ridge(data):
    # Uncompressed tar is seekable. Read a single small member into memory.
    with tarfile.open(data/'runs/archive_n054v3_n057_runs.tar', 'r:') as tar:
        blob = tar.extractfile('n054v3_batch1/A0/baseline_teacher.npz').read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != 'b9ab00ad8de6f64e7cb67ca5fa0d1bb1a3d6cfcaa173eb525cb32b49e668ba40':
        raise ValueError('N054 archived teacher member changed')
    with np.load(io.BytesIO(blob)) as f:
        return f['mu'], f['sd'], f['weight'], digest


def replay_markers(feats, data, t):
    """TT6c's per-marker ridge on the stored, stained split-0 rows only."""
    with np.load(data/'targets_v4c.npz') as f:
        observed, labels = f['mask'][:, :12], f['Y'][:, :12]
    eligible = (t['hist_row'] < 0) & (t['split'] == 0)
    rows = np.flatnonzero(eligible & observed.any(axis=1))
    if np.isin(t['slide'][rows], np.unique(t['slide'][t['hist_row'] >= 0])).any():
        raise ValueError('unexpected marker training rows on a GS slide')
    # Per-feature sample SD and unpenalized intercept as in train_gs_bind.py.
    total, square = np.zeros(512), np.zeros(512)
    for start in range(0, len(rows), 4096):
        block = feats[rows[start:start+4096]].astype(np.float64)
        total += block.sum(0)
        square += (block * block).sum(0)
    mu = total / len(rows)
    sd = np.sqrt(np.maximum(square - total * mu, 0) / (len(rows)-1)) + 1e-6
    weight = np.empty((513, 12))
    for j in range(12):
        r = np.random.default_rng(j).permutation(np.flatnonzero(eligible & observed[:, j]))[:60000]
        x = (feats[r] - mu) / sd
        model = Ridge(alpha=10, solver='cholesky').fit(x, labels[r, j])
        weight[:-1, j], weight[-1, j] = model.coef_, model.intercept_
    return mu, sd, weight


def routing_head(checkpoint, features, data, t):
    state = checkpoint['m']
    keys = ('c3.weight', 'c3.bias')
    if any(k in state for k in keys) and not all(k in state for k in keys):
        raise ValueError('incomplete cls3 checkpoint')
    if all(k in state for k in keys) and checkpoint['cfg']['w_cls'] > 0:
        w, b = (state[k].numpy() for k in keys)
        if w.shape != (3, features.shape[1]) or b.shape != (3,):
            raise ValueError('unexpected cls3 head dimensions')
        return (w, b), dict(source='checkpoint c3')

    with np.load(data/'targets_v4c.npz') as f:
        labels, pooled = f['cls3'], f['mask'][:, 8:12].any(axis=1)
    eligible = (t['batch'] == 0) & (t['split'] == 0) & (t['hist_row'] < 0) & pooled & (labels >= 0)
    rows = np.flatnonzero(eligible)
    if (not np.array_equal(np.unique(labels[rows]), np.arange(3)) or
            np.isin(t['slide'][rows], np.unique(t['slide'][t['hist_row'] >= 0])).any()):
        raise ValueError('invalid draw-0 pooled cls3 training rows')
    x = features[rows].astype(np.float64)
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=.1, max_iter=1000, random_state=0).fit(scaler.transform(x), labels[rows])
    w = clf.coef_ / scaler.scale_
    return (w, clf.intercept_ - w @ scaler.mean_), dict(source='draw0 pooled split0 cls3', n=len(rows))


class Teacher:
    def __init__(self, pair, data, t, archive):
        name, args = pair['teacher'], pair['args']
        if hashlib.sha256((Path(args['out'])/'result.json').read_bytes()).hexdigest() != pair['config_sha']:
            raise ValueError(f'configuration drift for {name}')
        if (args['targets'], args['aux_targets'], args['aux_names']) != ('targets_v4c.npz', 'targets_v6c.npz', 'autofl'):
            raise ValueError('unexpected TT6c target files')
        primary, aux = [Path(args[k]) for k in ('lf_feats', 'aux_feats')]
        if aux != data/'runs/n029_r18_v6c_fov/probe':
            raise ValueError('N054 frozen autofl ridge requires the registered auxiliary encoder')
        for path, target, seed in ((primary, args['targets'], int(name[1:])), (aux, args['aux_targets'], 0)):
            cfg = json.loads((path.parent/'result.json').read_text())['args']
            if (cfg['protocol'], cfg['include_gs'], cfg['targets'], cfg['seed']) != ('fov', 0, target, seed):
                raise ValueError(f'unexpected LF encoder training configuration: {path}')
        self.x, self.aux = [np.load(path/'lf_feats_all.npy', mmap_mode='r') for path in (primary, aux)]
        if self.x.shape != self.aux.shape or self.x.shape != (len(t['split']), 512):
            raise ValueError('LF feature schema mismatch')
        for path in (primary, aux):
            ids = np.load(path/'lf_feats_hist_indices.npy')
            hist = np.load(path/'lf_feats_hist_hist_row.npy')
            if not np.array_equal(t['hist_row'][ids], hist) or not np.array_equal(np.sort(hist), np.arange(15712)):
                raise ValueError('LF global/histology index mismatch')
        ck = torch.load(primary.parent/'ckpt.pt', map_location='cpu', weights_only=False)
        self.head, class_source = routing_head(ck, self.x, data, t)
        mu, sd, w, digest = archive
        if name == 't0':
            if primary != data/'runs/n029_r18_v4c_fov/probe':
                raise ValueError('unexpected t0 feature source for N054')
            self.primary = mu[:512], sd[:512], np.vstack((w[:512, :12], w[-1, :12]))
        else:
            self.primary = replay_markers(self.x, data, t)
        self.secondary = mu[512:], sd[512:], np.r_[w[512:-1, -1], w[-1, -1]]
        # Parity control against stored TT6c outputs, with no measured labels.
        with np.load(Path(args['out'])/'test_preds.npz') as ref:
            i = np.linspace(0, len(ref['gidx']) - 1, 16, dtype=int)
            g = ref['gidx'][i]
            if not np.array_equal(ref['rows'][i], t['hist_row'][g]):
                raise ValueError('reference teacher row mismatch')
            actual, _ = self.predict(g)
            err = abs(actual - ref['teacher'][i]).max(axis=0)
        if not np.isfinite(err).all() or err.max() > .02:
            raise ValueError(f'LF ridge disagrees with saved TT6c teacher: {err}')
        self.audit = dict(teacher=name, lf_class=class_source, config_sha=pair['config_sha'], primary=str(primary), auxiliary=str(aux),
                          archive_member_sha256=digest, parity_n=len(i), max_absolute_error=err.tolist())

    def class_probabilities(self, rows):
        x = self.x[rows].astype(np.float32)
        logits = x @ self.head[0].T + self.head[1]
        if not np.isfinite(logits).all():
            raise ValueError('nonfinite cls3 logits')
        return softmax(logits.astype(np.float64), axis=1)

    def predict(self, rows):
        marker, routes = [], []
        mu, sd, w = self.primary
        mu_a, sd_a, w_a = self.secondary
        for start in range(0, len(rows), 4096):
            r = rows[start:start+4096]
            x = self.x[r].astype(np.float32)
            p = ((x-mu)/sd) @ w[:-1] + w[-1]
            af = ((self.aux[r].astype(np.float32)-mu_a)/sd_a) @ w_a[:-1] + w_a[-1]
            marker.append(np.column_stack((p, af)))
            routes.append((x @ self.head[0].T + self.head[1]).argmax(axis=1))
        p = np.concatenate(marker)
        if not np.isfinite(p).all():
            raise ValueError('nonfinite teacher predictions')
        return p, np.concatenate(routes)


class PBC:
    def __init__(self, data, harness):
        path = data/'external/pbc/PBC_dataset_normal_DIB.zip'
        with zipfile.ZipFile(path) as z:
            names = harness.sorted_names(z, ('.jpg',))
        self.row = {name: i for i, name in enumerate(names)}
        self.features = np.load(data/'emb/pbc_dinobloom_B.npy', mmap_mode='r')
        self.labels = np.load(data/'emb/pbc_labels.npy', mmap_mode='r')
        self.classes = json.loads((data/'emb/pbc_classes.json').read_text())
        if self.features.shape != (len(names), 768) or self.labels.shape != (len(names),):
            raise ValueError('PBC N008 feature/manifest length mismatch')

    def get(self, records):
        rows = np.array([self.row[name] for _, name, _ in records])
        cls = [r[2] for r in records]
        if [self.classes[i] for i in self.labels[rows]] != cls:
            raise ValueError('PBC N008 label order differs from archive order')
        return self.features[rows]


def limit_per_class(records, n, seed):
    if n is None:
        return records
    classes = np.array([r[2] for r in records])
    rng = np.random.default_rng(seed)
    rows = np.sort(np.concatenate([sample(np.flatnonzero(classes == c), n, rng) for c in np.unique(classes)]))
    return [records[i] for i in rows]


@torch.inference_mode()
def embed_mll(data, records, batch_size=16):
    import timm
    model = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=0, img_size=224)
    ck = torch.load(data/'models/DinoBloom-B.pth', map_location='cpu', weights_only=True)['teacher']
    model.load_state_dict({k[9:]: v for k, v in ck.items() if k.startswith('backbone.') and k != 'backbone.mask_token'})
    del ck
    model.eval()
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    sd = torch.tensor([.229, .224, .225])[:, None, None]
    features = []
    with ExitStack() as stack:
        archives = {p: stack.enter_context(zipfile.ZipFile(p)) for p in sorted({p for p, _, _ in records})}
        for start in range(0, len(records), batch_size):
            batch = []
            for path, name, _ in records[start:start+batch_size]:
                with Image.open(io.BytesIO(archives[path].read(name))) as im:
                    x = torch.from_numpy(np.asarray(im.convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1)
                batch.append(F.interpolate(x[None], size=224, mode='bilinear', antialias=True)[0])
            features.append(model((torch.stack(batch)-mean)/sd).numpy())
    return np.concatenate(features)
