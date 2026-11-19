"""N049 archive validation and masked probabilistic supervision (no LF inputs)."""
import hashlib
import json
import numpy as np


def source_digest(T, rows, cols):
    """Pin source labels/metadata only; never use held-out-draw Y in a fit."""
    h = hashlib.sha256()
    arrays = [np.asarray(rows, dtype='<i8'), np.asarray(T['names'])[cols].astype('U'),
              np.asarray(T['mask'])[rows][:, cols].astype(bool),
              np.where(np.asarray(T['mask'])[rows][:, cols],
                       np.asarray(T['Y'])[rows][:, cols], 0).astype('<f8')]
    arrays += [np.asarray(T[k])[rows].astype('<i8') for k in ('split', 'batch', 'hist_row')]
    for v in arrays:
        h.update(str(v.shape).encode()); h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def load_pospos(path, T, tr, protocol, include_gs=0):
    """Fail closed on row/schema/mask/source-label mismatch; expand to global rows."""
    if protocol not in ('b0to1', 'b1to0') or include_gs:
        raise ValueError('--pospos requires a single training draw and --include-gs 0')
    source = 0 if protocol == 'b0to1' else 1
    cols = np.array([i for i, n in enumerate(T['names']) if str(n).startswith('single:')])
    if not len(cols): raise ValueError('no single-marker columns')
    expected = np.flatnonzero((T['batch'] == source) & (T['hist_row'] < 0))
    with np.load(path, allow_pickle=False) as z:
        g, p, m = z['G'], z['P'], z['mask']
        meta = json.loads(str(z['metadata'].item()))
        if g.dtype.kind not in 'iu' or g.ndim != 1 or not np.array_equal(g, expected):
            raise ValueError('pospos G must be all clean source-draw rows, sorted and unique')
        if not np.isin(tr, g).all(): raise ValueError('pospos does not cover training rows')
        if list(z['names'].astype(str)) != list(np.asarray(T['names'])[cols].astype(str)):
            raise ValueError('pospos marker schema differs')
        if m.dtype != np.bool_ or m.shape != (len(g), len(cols)) or p.shape != m.shape:
            raise ValueError('pospos shape or mask dtype invalid')
        if not np.array_equal(m, T['mask'][g][:, cols]):
            raise ValueError('pospos mask differs from curated single-marker mask')
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or (p[~m] != 0).any():
            raise ValueError('pospos probabilities invalid or nonzero on missing labels')
        if not np.array_equal(z['fit_mask'], m & (T['split'][g, None] == 0)):
            raise ValueError('pospos fit mask must be clean source split-0 only')
        digest = source_digest(T, g, cols)
        if (meta.get('schema') != 'n049-v1' or meta.get('source_draw') != source
                or meta.get('source_sha256') != digest):
            raise ValueError('pospos provenance/source labels differ')
        py = np.zeros((len(T['Y']), len(cols)), np.float32)
        pm = np.zeros_like(py, dtype=bool)
        py[g], pm[g] = p, m
    with open(path, 'rb') as f: file_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    return py, pm, {'sha256': file_hash, 'source_sha256': digest,
                    'source_draw': source, 'names': list(np.asarray(T['names'])[cols].astype(str))}


def add_pos_head(model, n_features, n_markers):
    """Called ONLY when opted in, after legacy modules: defaults consume no RNG."""
    from torch import nn
    model['pos'] = nn.Linear(n_features, n_markers)


def masked_soft_bce(logits, targets, mask):
    """Mean per present marker, then over markers (like MSE); select BEFORE BCE."""
    import torch
    import torch.nn.functional as F
    mask = mask.bool()
    if not mask.any(): return logits[mask].sum()
    terms = F.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction='none')
    by_marker = torch.zeros(logits.shape[1], device=logits.device, dtype=terms.dtype)
    by_marker.scatter_add_(0, mask.nonzero(as_tuple=True)[1], terms)
    count = mask.sum(0)
    return (by_marker[count > 0] / count[count > 0]).mean()
