"""CPU mathematical/invariance checks for N060--N062 (no GPU or mock encoder)."""
import ast
import itertools
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bsccm_heads import (HLGauss, TwoPart, PanelHead, TrainAdapters,
                         FOVBatchSampler, masked_mean, pair_difference_loss)

torch.manual_seed(7)
rng = np.random.default_rng(7)
names = ['single:CD45', 'single:CD123', 'single:CD19', 'single:CD56', 'single:CD14', 'single:CD16', 'single:HLA-DR', 'single:CD3',
         'grp:CD123/HLA-DR/CD14', 'grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16', 'autofl']
y = rng.normal(size=(100, 13)).astype('float32')
m = rng.uniform(size=y.shape) > .5
m[:, 8] = False  # source batch-1 unsupported pooled target
h = torch.randn(100, 9, requires_grad=True)
yt, mt = torch.tensor(y), torch.tensor(m)
w, tw = torch.ones_like(yt), torch.ones(13)
# Legacy masking/normalization, including an entirely unobserved target.
p = torch.randn_like(yt, requires_grad=True)
legacy = ((p - yt).square() * mt).sum(0) / mt.sum(0).clamp_min(1e-6)
assert torch.allclose(masked_mean((p-yt).square(), mt, w, tw), legacy[mt.any(0)].mean())
tw[-1] = 0
loss = masked_mean((p-yt).square(), mt, w, tw)
loss.backward(); assert torch.count_nonzero(p.grad[:, -1]) == 0
assert torch.count_nonzero(p.grad[:, 8]) == 0
assert masked_mean(p.square(), mt & False, w, tw) == 0
print('PASS masked regression equals legacy; missing/drop-autofl gradients zero')

hl = HLGauss(9, y, m)
q = hl.target_probs(yt)
assert torch.allclose(q.sum(-1), torch.ones_like(yt), atol=1e-6)
lo, hi = np.quantile(y[m[:, 0], 0], [.001, .999]); span = hi-lo
assert np.allclose(hl.edges[0, [0, -1]], [lo-.1*span, hi+.1*span])
extreme = yt.clone(); extreme[:, 1] = 1e20
assert torch.isfinite(hl.target_probs(extreme)).all()
pred = hl(h)
assert ((pred >= hl.centres[:, 0]) & (pred <= hl.centres[:, -1])).all()
loss = masked_mean(hl.cell_loss(h, yt) + .1*(pred-yt).square(), mt, w, tw)
loss.backward(); assert torch.isfinite(hl.linear.weight.grad).all()
print('PASS Gaussian bin integrals normalize, correct quantiles/expectation, extreme tails, gradients')

hp = TwoPart(9, names, y, m, .98)
for i, j in enumerate(hp.indices):
    values = y[m[:, j], j]
    assert np.isclose(hp.thresholds[i], np.quantile(values, .98))
    yy = torch.tensor(values) > hp.thresholds[i]
    if hp.gate_active[i]:
        assert torch.allclose(hp.class_weight[i, 0]*(~yy).sum(), hp.class_weight[i, 1]*yy.sum())
v = hp.components(h)
assert torch.allclose(hp(h)[:, hp.indices], v[..., 0].sigmoid()*v[..., 2] + (1-v[..., 0].sigmoid())*v[..., 1])
gate = hp.gate_loss(h, yt, mt, w, tw)
manual = torch.zeros_like(yt)
for i, j in enumerate(hp.indices):
    label = yt[:, j] > hp.thresholds[i]
    cw = torch.where(label, hp.class_weight[i, 1], hp.class_weight[i, 0])
    manual[:, j] = F.binary_cross_entropy_with_logits(hp.components(h)[:, i, 0], label.float(), reduction='none') * cw * hp.gate_active[i]
assert torch.allclose(gate, masked_mean(manual, mt, w, tw))
gate.backward(); assert torch.isfinite(hp.parts.weight.grad).all()
print('PASS two-part strict source thresholds, class balance, gate/amplitude equation, backward')

fov = torch.tensor([0, 0, 0, 1, 1, 2]); rows = torch.tensor([0, 1, 1, 3, 4, 5])
pp, yy = p.detach()[:6], yt[:6]
mm = mt[:6]
actual = pair_difference_loss(pp, yy, mm, fov, tw, rows)
per = []
for j in range(13):
    vv = [((pp[i,j]-pp[k,j])-(yy[i,j]-yy[k,j])).square()
          for i, k in itertools.combinations(range(6), 2) if fov[i]==fov[k] and rows[i]!=rows[k] and mm[i,j] and mm[k,j]]
    if vv and tw[j]: per.append(torch.stack(vv).mean())
assert torch.allclose(actual, torch.stack(per).mean())
shift = torch.tensor([12., 12., 12., -7., -7., 2.])[:, None]
assert torch.allclose(actual, pair_difference_loss(pp, yy+shift, mm, fov, tw, rows), atol=2e-5)
assert pair_difference_loss(pp, yy, mm & False, fov, tw, rows) == 0
sam = FOVBatchSampler(np.repeat(np.arange(5), 4), np.zeros(20), 8, 2, 9)
batches = list(sam)
for batch in batches:
    _, count = np.unique(np.array(batch)//4, return_counts=True)
    assert np.array_equal(count, [4, 4])
assert batches == list(sam)
sam.epoch = 1; assert batches != list(sam)
print('PASS pair loss matches explicit unordered pairs, cancels FOV offsets, excludes duplicates; FOV sampler')

adapt = TrainAdapters(np.array([1, 1, 2]), np.array([3, 3, 4]), 13, True, True)
pp = torch.randn(3, 13, requires_grad=True); l3 = torch.randn(3, 3, requires_grad=True)
fi, si = torch.tensor(adapt.fi), torch.tensor(adapt.si)
assert torch.equal(adapt(pp, l3, fi, si), pp)
assert adapt.penalty(.01, 1, tw) == 0
with torch.no_grad():
    adapt.offset.fill_(2); adapt.delta.fill_(.3); adapt.shift.fill_(.1)
qq = adapt(pp, l3, fi, si)
assert torch.allclose(qq, 1.3*pp + 2.1)
qq.sum().backward(); assert l3.grad is None
try: adapt(pp, l3, fi-100, si)
except ValueError: pass
else: raise AssertionError('non-source rows accepted')
print('PASS identity/shrunk training adapters, predicted-class detach, non-source rejection')

panel = PanelHead(9, names)
assert torch.all(F.softplus(panel.single_slope) > 0)
assert torch.equal(panel.design*F.softplus(panel.raw_w), panel.design)
ph = torch.randn(3, 9, requires_grad=True)
# Pooled-only supervision reaches the same marker factors used by singles.
pr = panel(ph); pr[:, 9].square().mean().backward()
for j in (2, 3, 7): assert panel.z.weight.grad[j].abs().sum() > 0
assert panel.z.weight.grad[0].abs().sum() == 0
print('PASS panel nonnegative design initialization, positive links, pooled-to-single factor gradients')

orig = Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py').read_text()
new = Path(__file__).with_name('train_lf.py').read_text()
def fn(src, name):
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return '\n'.join(src.splitlines()[node.lineno-2:node.end_lineno])
assert fn(orig, 'evaluate') == fn(new, 'evaluate')
assert "model['readout']" not in fn(new, 'fwd')
print('PASS evaluate() byte-identical; fwd() has no training adapters')
print('ALL MATHEMATICAL / INVARIANCE CHECKS PASSED')
