"""Supervised LF -> markers (+ cell class) baseline on curated targets_v2 (N005/N006).
Masked MSE over 13 targets (8 single-antibody markers, 4 pooled 'all' groups, autofluorescence) + CE on cls3 (and cls10).
Protocols: fov (train split 0 / eval split 2, both batches) | b1to0 (train batch 1, eval batch-0 test FOVs) | b0to1.
Eval: per-target Pearson; within-class Pearson for the 4 pooled groups (batch-0 'all' cells have gates); class-oracle baseline
(r of the true per-class mean) so 'beyond cell type' signal is visible; cls3 accuracy.
usage: train_lf.py --out DIR [--protocol fov] [--channels all] [--phase 1] [--arch resnet18] [--epochs 8]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from bsccm.data import chan_list, W
from bsccm.gpu_norm import RawSet, GPUNorm
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--protocol', default='fov'); ap.add_argument('--channels', default='all')
ap.add_argument('--phase', type=int, default=1); ap.add_argument('--arch', default='resnet18'); ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--workers', type=int, default=15)
ap.add_argument('--max-train', type=int, default=0); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--w-cls', type=float, default=0.5); ap.add_argument('--include-gs', type=int, default=0)
ap.add_argument('--init-ssl', default=''); ap.add_argument('--tail-weight', type=float, default=0.0); ap.add_argument('--focal-c10', type=float, default=0.0); ap.add_argument('--targets', default='targets_v4.npz'); ap.add_argument('--dann', type=float, default=0.0)
ap.add_argument('--stem', default='std', choices=['std', 'nomaxpool']); ap.add_argument('--zoom', type=int, default=1)   # N026 resolution: centre crop 128/zoom -> upsample to 128
ap.add_argument('--phys-aug', type=float, default=0.0)   # N041: probability of raw-count instrument perturbation (off)
ap.add_argument('--consist', type=float, default=0.0)   # N041: MSE weight between independently perturbed prediction views (off)
ap.add_argument('--pseudo', default='')   # N040: comma-separated teacher train_preds.npz (P,G on TRAINING-draw rows), averaged; pseudo-label targets a cell lacks
ap.add_argument('--pseudo-w', type=float, default=0.3)   # N040: weight of the pseudo-label MSE (only used with --pseudo)
ap.add_argument('--dump-train-preds', type=int, default=0)   # N040: load the finished checkpoint, predict all training rows -> <out>/train_preds.npz, exit
ap.add_argument('--lowpass', type=float, default=0.0)   # N042 probe: Gaussian low-pass (sigma px) on the normalised LF input, train AND eval (0 = off)
ap.add_argument('--dual-zoom', type=int, default=1)   # N026: shared encoder sees full view AND centre-zoom view; features concatenated
ap.add_argument('--adabn', type=int, default=0)   # N026: after training, recompute BN stats on N unlabelled target-batch images (split 0), evaluate again
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
if not np.isfinite(a.phys_aug) or not 0 <= a.phys_aug <= 1: ap.error('--phys-aug must be in [0,1]')
if not np.isfinite(a.consist) or a.consist < 0: ap.error('--consist must be nonnegative')
if a.consist > 0 and a.phys_aug == 0: ap.error('--consist requires --phys-aug > 0')
PHYS = a.phys_aug > 0
if PHYS:
    if a.protocol not in ('b1to0', 'b0to1'): ap.error('N041 requires a single source draw: --protocol b1to0 or b0to1')
    if a.include_gs or a.dann > 0 or a.adabn: ap.error('N041 source-only requires --include-gs 0 --dann 0 --adabn 0')
T = np.load(W + '/' + a.targets); names = list(T['names']); K = len(names)
Y, Mk, c3, c10, sp, b = T['Y'], T['mask'], T['cls3'], T['cls10'], T['split'], T['batch']
has = Mk.any(1) | (c3 >= 0)
if not a.include_gs: has &= T['hist_row'] < 0   # keep the 15,712 GS cells (label: autofl only) out of LF training -> clean GS autofl test
if a.protocol == 'fov': tr = (sp == 0) & has; ev = (sp == 2) & has
elif a.protocol == 'b1to0': tr = (b == 1) & has; ev = (b == 0) & (sp == 2) & has
elif a.protocol == 'b0to1': tr = (b == 0) & has; ev = (b == 1) & (sp == 2) & has
else: raise SystemExit('protocol?')
tr, ev = np.where(tr)[0], np.where(ev)[0]
if a.max_train: tr = np.random.default_rng(0).permutation(tr)[:a.max_train]
print(f'train {len(tr)} eval {len(ev)} protocol {a.protocol}', flush=True)
nin = len(chan_list(a.channels)) + a.phase
Wt_ = np.ones(Y.shape, np.float32)
if a.tail_weight > 0:   # fine-grained: up-weight rare marker values (density^-alpha in 60 bins over [-4, 4], per batch x target, clipped [.2, 10], mean 1)
    edges = np.linspace(-4, 4, 61)
    for j in range(K):
        for bb in (0, 1):
            m_tr = np.zeros(len(Y), bool); m_tr[tr] = True; sel = m_tr & Mk[:, j] & (b == bb)
            if sel.sum() < 100: continue
            h, _ = np.histogram(np.clip(Y[sel, j], -4, 4), edges); dens = (h + 1) / (h.sum() + len(h))
            allsel = Mk[:, j] & (b == bb); idx = np.clip(np.digitize(np.clip(Y[allsel, j], -4, 4), edges) - 1, 0, 59)
            w = np.clip(dens[idx] ** (-a.tail_weight), 0.2, 10.0); w = w / w[m_tr[allsel]].mean(); Wt_[allsel, j] = w
LAB = {'Y': Y, 'M': Mk, 'c3': c3.astype(np.int64), 'c10': c10.astype(np.int64), 'W': Wt_}
PSEUDO_ID = None
if a.pseudo:   # N040 noisy student: teacher predictions as extra targets ONLY where the cell has no real label; training draw only
    import hashlib
    if not np.isfinite(a.pseudo_w) or a.pseudo_w < 0: raise SystemExit('--pseudo-w must be finite and >= 0')
    _Ps = [np.load(f) for f in a.pseudo.split(',')]; _G0 = _Ps[0]['G']
    for q in _Ps:   # red-team N040 F4: integrity checks (explicit, not asserts)
        if q['G'].ndim != 1 or not np.issubdtype(q['G'].dtype, np.integer) or len(np.unique(q['G'])) != len(q['G']) or q['G'].min() < 0 or q['G'].max() >= len(Y): raise SystemExit('teacher G invalid')
        if not np.array_equal(q['G'], _G0): raise SystemExit('teacher row sets differ')
        if q['P'].shape != (len(_G0), K) or not np.isfinite(q['P']).all(): raise SystemExit('teacher P has wrong shape or non-finite values')
        if 'names' in q and [str(n) for n in q['names']] != [str(n) for n in names]: raise SystemExit('teacher target schema differs')
    if not (np.isin(tr, _G0).all() and len(np.unique(b[_G0])) == 1 and b[_G0[0]] == b[tr[0]] and (T['hist_row'][_G0] < 0).all()): raise SystemExit('pseudo rows must cover tr, one (training) draw, no Giemsa')
    _cols = np.array([(n.startswith('single:') or n.startswith('grp:')) and 'CD123/HLA' not in n for n in names]) & (Mk[tr].sum(0) >= 1000)   # F2: common-11 targets with real source support only
    PY = np.zeros(Y.shape, np.float32); PM = np.zeros(Y.shape, bool); PY[_G0] = np.mean([q['P'] for q in _Ps], 0); PM[_G0] = True; PM &= ~Mk; PM[:, ~_cols] = False
    PY[~PM] = 0.0   # F4: nothing but masked-in values can enter the loss
    PSEUDO_ID = dict(w=float(a.pseudo_w), cols=[names[k] for k in np.flatnonzero(_cols)],   # F3: pin teacher CONTENT, not paths
                     sha256=[hashlib.sha256(np.ascontiguousarray(q['P']).tobytes() + np.ascontiguousarray(q['G']).tobytes()).hexdigest() for q in _Ps])
    LAB['PY'] = PY; LAB['PM'] = PM; print(f'N040 pseudo labels: {len(_Ps)} teachers, cols {PSEUDO_ID["cols"]}, {int(PM[tr].sum())} cell x target entries on {len(tr)} training cells', flush=True)
mk = lambda rows, train: torch.utils.data.DataLoader(RawSet(rows, LAB), batch_size=a.bs, shuffle=train, drop_last=train,
                                                      num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=8)
phys_aug = None; phys_calibration = None
if PHYS:
    from bsccm.phys_aug import PhysAug, calibrate_source_borders
    _calf = a.out + '/phys_aug_calibration.json'
    if os.path.exists(_calf):   # red-team m4: a resumed run reuses its stored calibration (no float re-derivation on another CPU type)
        phys_calibration = json.load(open(_calf)); print('N041: reusing', _calf, flush=True)
    else:
        cal_rows = tr[(sp[tr] == 0) & (T['hist_row'][tr] < 0)]
        phys_calibration = calibrate_source_borders(np.load(W + '/pack/led_array.npy', mmap_mode='r'),
                                                   cal_rows, b, sp, T['hist_row'])
        from bsccm.phys_aug import family_gain_log_sd   # M1: brightness jitter sized from the source draw's normalised border variation
        _gn, _led, _dpc = GPUNorm('cpu'), np.load(W + '/pack/led_array.npy', mmap_mode='r'), np.load(W + '/pack/dpc.npy', mmap_mode='r')
        _rows = np.asarray(phys_calibration['rows']); _bm = np.ones((128, 128), bool); _bm[8:-8, 8:-8] = False; _dc = []
        for _ch in np.array_split(_rows, max(1, len(_rows) // 64)):
            _x = _gn(torch.from_numpy(np.stack([np.array(_led[i]).view(np.int16) for i in _ch])), torch.from_numpy(np.stack([np.array(_dpc[i]) for i in _ch])),
                     torch.from_numpy(_ch), 'all', False); _dc.append(_x[:, :, torch.from_numpy(_bm)].mean(-1).numpy())
        phys_calibration['gain_log_sd'] = family_gain_log_sd(np.concatenate(_dc), _gn.s_c.numpy(), mult=2.)
        phys_calibration['gain_log_sd_method'] = 'family MIN of 2 x robust sd of normalised 8-px border mean x s_c (source split-0 calibration rows)'
    phys_aug = PhysAug(a.phys_aug, phys_calibration['gain_adu_per_e'], phys_calibration['offset_sd_adu'], device='cuda',
                       gain_log_sd=phys_calibration['gain_log_sd'])
    print('N041 source-border calibration:', {k: phys_calibration[k] for k in ('source_draw', 'gain_adu_per_e', 'offset_sd_adu', 'gain_log_sd', 'rows_sha256')}, flush=True)
gnorm = GPUNorm('cuda'); CH = chan_list(a.channels)
_cnt = np.bincount(c10[tr][c10[tr] >= 0], minlength=10).astype(np.float32); C10W = torch.tensor(1 / np.sqrt(np.maximum(_cnt, 1)), device='cuda'); C10W = C10W / C10W[_cnt > 0].mean()
def prep(raw, ph, g, train):   # raw int16 records -> normalised LF on GPU (+-4 px translation when training)
    raw = raw.cuda(non_blocking=True)
    if train and phys_aug is not None: raw = phys_aug(raw)   # raw counts BEFORE flat-field/slide/log normalisation; phase untouched
    x = gnorm(raw, ph.cuda(non_blocking=True), g.cuda(non_blocking=True), CH, bool(a.phase), shift=4 if train else 0)
    if a.lowpass > 0:   # depthwise separable Gaussian, reflect padding; applied before zoom
        r_ = int(np.ceil(3 * a.lowpass)); k_ = torch.exp(-0.5 * (torch.arange(-r_, r_ + 1, device=x.device, dtype=x.dtype) / a.lowpass) ** 2); k_ = k_ / k_.sum()
        c_ = x.shape[1]; x = F.pad(x, (r_, r_, r_, r_), mode='reflect')
        x = F.conv2d(x, k_.view(1, 1, 1, -1).repeat(c_, 1, 1, 1), groups=c_); x = F.conv2d(x, k_.view(1, 1, -1, 1).repeat(c_, 1, 1, 1), groups=c_)
    if a.zoom > 1:
        s_ = 128 // a.zoom; o = (128 - s_) // 2; x = F.interpolate(x[..., o:o + s_, o:o + s_], size=(128, 128), mode='bilinear', align_corners=False)
    return x
dtr, dev = mk(tr, True), mk(ev, False)
if a.dann > 0:
    assert a.protocol in ('b1to0', 'b0to1'), 'dann: batch hold-out protocols only'
    tb = 0 if a.protocol == 'b1to0' else 1; tgt = np.where((b == tb) & (sp == 0))[0]      # unlabelled target images, training FOVs only
    def _forever(dl):
        while True:
            for z in dl: yield z
    it_tgt = _forever(mk(tgt, True)); print(f'DANN: {len(tgt)} unlabelled target-batch images (split 0 only)', flush=True)
SSL = bool(a.init_ssl); net = timm.create_model(a.arch, pretrained=False, num_classes=0, in_chans=48 if SSL else nin)
if SSL:
    assert a.channels == 'all' and a.phase == 1, 'SSL init expects all 23 LED + phase'
    net.load_state_dict(torch.load(a.init_ssl, map_location='cpu', weights_only=True), strict=True); print('init from SSL', a.init_ssl, flush=True)
if a.stem == 'nomaxpool': net.maxpool = nn.Identity()   # stride 2 instead of 4 before layer1 (weights unchanged)
NF = net.num_features * (2 if a.dual_zoom > 1 else 1)
model = nn.ModuleDict({'enc': net, 'reg': nn.Linear(NF, K), 'c3': nn.Linear(NF, 3), 'c10': nn.Linear(NF, 10),
                      'dom': nn.Sequential(nn.Linear(NF, 256), nn.ReLU(), nn.Linear(256, 1))}).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps = a.epochs * len(dtr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None
def fwd(x):
    if SSL: x = torch.cat([x, torch.ones_like(x)], 1)
    with torch.autocast('cuda', torch.bfloat16):
        h = model['enc'](x.contiguous(memory_format=torch.channels_last))
        if a.dual_zoom > 1:   # second view through the SAME encoder: centre crop 128/dual_zoom upsampled to 128
            s_ = 128 // a.dual_zoom; o = (128 - s_) // 2; xz = F.interpolate(x[..., o:o + s_, o:o + s_], size=(128, 128), mode='bilinear', align_corners=False)
            h = torch.cat([h, model['enc'](xz.contiguous(memory_format=torch.channels_last))], 1)
    h = h.float(); return model['reg'](h), model['c3'](h), model['c10'](h), h
def pearson(p, y):
    p, y = p - p.mean(), y - y.mean(); return float((p * y).sum() / (np.sqrt((p * p).sum() * (y * y).sum()) + 1e-9))
@torch.no_grad()
def evaluate(save=False, tag=''):
    model.eval(); P, C3, G = [], [], []
    C10 = []
    for raw, ph, g, y, m, k3, k10, _w, *_ in dev:
        p, l3, l10, h = fwd(prep(raw, ph, g, False)); P.append(p.cpu().numpy()); C3.append(l3.argmax(1).cpu().numpy()); C10.append(l10.argmax(1).cpu().numpy()); G.append(g.numpy())
    P, C3, G = np.concatenate(P), np.concatenate(C3), np.concatenate(G); Yt, Mt, k3 = Y[G], Mk[G], c3[G]; r = {}
    for j, n in enumerate(names):
        v = Mt[:, j]
        if v.sum() > 20: r[f'r/{n}'] = pearson(P[v, j], Yt[v, j]); r[f'n/{n}'] = int(v.sum())
    lab = k3 >= 0
    if lab.any():
        r['cls3_acc'] = float((C3[lab] == k3[lab]).mean())
        for j in range(8, 12):   # pooled groups on gated cells: within-class r + class-oracle baseline
            v = lab & Mt[:, j]
            if v.sum() < 50: continue
            mean_by = {k: Yt[v & (k3 == k), j].mean() for k in range(3)}; oracle = np.array([mean_by[k] for k in k3[v]])
            r[f'oracle_classmean_r/{names[j]}'] = pearson(oracle, Yt[v, j]); r[f'r_gated/{names[j]}'] = pearson(P[v, j], Yt[v, j])
            for k, kn in enumerate(['lym', 'gran', 'mono']):
                vk = v & (k3 == k)
                if vk.sum() > 30: r[f'r_within_{kn}/{names[j]}'] = pearson(P[vk, j], Yt[vk, j])
    r['r_mean_markers'] = float(np.mean([r[f'r/{n}'] for n in names[:12] if f'r/{n}' in r]))
    trained = Mk[tr].sum(0) >= 1000   # v5 QC masks can leave a target without training labels (grp:CD123/HLA-DR/CD14 when training on batch 1)
    r['r_mean_markers_valid'] = float(np.mean([r[f'r/{n}'] for j, n in enumerate(names[:12]) if f'r/{n}' in r and trained[j]]))
    r['n_markers_valid'] = int(sum(1 for j, n in enumerate(names[:12]) if f'r/{n}' in r and trained[j]))
    wv = [v for k, v in r.items() if k.startswith('r_within_')]
    if wv: r['fine/within_class_r_mean_grp'] = float(np.mean(wv))
    k10 = c10[G]; l10m = k10 >= 0
    if l10m.sum() > 100:
        C10 = np.concatenate(C10); acc = [(C10[l10m & (k10 == c)] == c).mean() for c in range(10) if (l10m & (k10 == c)).sum() > 5]
        r['fine/cls10_balanced_acc'] = float(np.mean(acc)); r['fine/cls10_recall'] = {int(c): float((C10[l10m & (k10 == c)] == c).mean()) for c in range(10) if (l10m & (k10 == c)).sum() > 5}
        gran = np.isin(k10, [1, 2, 3, 5]); jc = names.index('grp:CD16')
        if (gran & (k10 == 5)).sum() > 5:
            from sklearn.metrics import roc_auc_score
            r['fine/eos_gate_auroc_from_predCD16'] = float(roc_auc_score(k10[gran] == 5, -P[gran, jc]))
    if save: np.savez(a.out + f'/eval_preds{tag}.npz', P=P, G=G, C3=C3)
    model.train(); return r
try:
    if a.dump_train_preds: raise RuntimeError('no wandb for a teacher dump (would append to the teacher run)')
    import wandb; run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), name=os.path.basename(a.out.rstrip('/')), id=os.path.basename(a.out.rstrip('/')), resume='allow', config=vars(a))
except Exception as e: print('wandb off:', e); run = None
_NEW_DEFAULTS = {'stem': 'std', 'zoom': 1, 'dual_zoom': 1, 'lowpass': 0.0, 'phys_aug': 0.0, 'consist': 0.0, 'pseudo': '', 'pseudo_w': 0.3}   # options added after older checkpoints were written: omitted at default so they still resume
def _cfg():
    cfg = {k: v for k, v in vars(a).items() if k not in ('workers', 'out', 'adabn', 'dump_train_preds') and not (k in _NEW_DEFAULTS and v == _NEW_DEFAULTS[k])}
    if PHYS: cfg['phys_calibration'] = phys_calibration   # pin source row IDs, fitted gain/offsets and method on resume
    if PSEUDO_ID is not None: cfg['pseudo_id'] = PSEUDO_ID   # N040 F3
    return cfg
ck = a.out + '/ckpt.pt'; ep0 = 0; log = []
if os.path.exists(ck):
    s = torch.load(ck, map_location='cuda', weights_only=False)
    cfg = _cfg()
    if s.get('cfg') != cfg or not np.array_equal(s.get('tr'), tr) or not np.array_equal(s.get('ev'), ev):
        raise SystemExit(f'refusing to resume {ck}: config/rows differ from the checkpoint (red-team F3)')
    model.load_state_dict(s['m']); opt.load_state_dict(s['o']); sched.load_state_dict(s['s']); ep0, log = s['ep'], s['log']; print('resumed', ep0, flush=True)
if PHYS and not a.dump_train_preds:
    with open(a.out + '/phys_aug_calibration.json', 'w') as f: json.dump(phys_calibration, f, indent=2)
if a.dump_train_preds:   # N040 teacher pass: needs the finished checkpoint (resumed above); writes P,G for every training row, then exits
    if not os.path.exists(ck) or ep0 != a.epochs: raise SystemExit('dump-train-preds needs a FINISHED checkpoint in --out (nothing written)')
    model.eval(); _PP, _GG = [], []
    with torch.no_grad():
        for raw, ph, g, *_ in mk(tr, False): _p = fwd(prep(raw, ph, g, False))[0]; _PP.append(_p.float().cpu().numpy()); _GG.append(g.numpy())
    np.savez(a.out + '/train_preds.npz', P=np.concatenate(_PP), G=np.concatenate(_GG), names=np.array([str(n) for n in names])); print('N040: wrote', a.out + '/train_preds.npz', len(tr), flush=True); sys.exit(0)
t0 = time.time(); step = ep0 * len(dtr)
for ep in range(ep0, a.epochs):
    tl = n = 0
    for raw, ph, g, y, m, k3, k10, wt, *pz in dtr:
        if a.consist > 0:   # transfer once; both prep calls independently sample raw-count nuisances and the existing translation
            raw, ph, g = raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True)
        y, m, k3, k10, wt = y.cuda(), m.cuda(), k3.cuda(), k10.cuda(), wt.cuda(); _last_h = fwd(prep(raw, ph, g, True)); p, l3, l10, _ = _last_h
        reg = (((p - y) ** 2) * m * wt).sum(0) / (m * wt).sum(0).clamp(min=1e-6); reg = reg[m.any(0)].mean()
        if (k3 >= 0).any():
            ce3 = F.cross_entropy(l3, k3, ignore_index=-1)
            if a.focal_c10 > 0:   # class-balanced focal loss on the 10 fine gates
                v10 = k10 >= 0; lp = F.log_softmax(l10[v10], 1); pt = lp.gather(1, k10[v10, None])[:, 0]
                ce10 = (-(C10W[k10[v10]]) * (1 - pt.exp()) ** a.focal_c10 * pt).mean() if v10.any() else p.sum() * 0
            else: ce10 = F.cross_entropy(l10, k10, ignore_index=-1)
            ce = ce3 + ce10
        else: ce = p.sum() * 0
        if a.dann > 0:
            rt, pt_, gt, *_ = next(it_tgt); _, _, _, ht = fwd(prep(rt, pt_, gt, True)); _, _, _, hs = _last_h
            hh = torch.cat([hs, ht]); dl_ = torch.cat([torch.zeros(len(hs)), torch.ones(len(ht))]).cuda()
            ce = ce + a.dann * F.binary_cross_entropy_with_logits(model['dom'](_GRL.apply(hh, 1.0))[:, 0], dl_)
        consistency = p.new_zeros(())
        if a.consist > 0:
            p2, _, _, _ = fwd(prep(raw, ph, g, True))
            consistency = F.mse_loss(p, p2)   # all K regression predictions; symmetric gradients, no labels or target-draw images
        loss = reg + a.w_cls * ce
        if a.pseudo:
            py, pm = pz[0].cuda(), pz[1].cuda().float()
            preg = (((p.float() - py) ** 2) * pm).sum(0) / pm.sum(0).clamp(min=1e-6); loss = loss + a.pseudo_w * (preg[pm.any(0)].mean() if pm.any() else p.sum() * 0)
        if a.consist > 0: loss = loss + a.consist * consistency
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        tl += loss.item(); n += 1; step += 1
        if step % 200 == 0:
            print(f'ep {ep} step {step}/{steps} loss {tl/n:.4f} {time.time()-t0:.0f}s', flush=True)
            if run: run.log({'train/loss': tl / n}, step=step)
    r = evaluate(save=(ep + 1 == a.epochs)); r.update(epoch=ep + 1, train_loss=tl / n, secs=time.time() - t0); log.append(r); print(json.dumps(r, default=float), flush=True)
    if run: run.log({f'eval/{k}': v for k, v in r.items() if not isinstance(v, dict)}, step=step)
    torch.save({'m': model.state_dict(), 'o': opt.state_dict(), 's': sched.state_dict(), 'ep': ep + 1, 'log': log, 'tr': tr, 'ev': ev,
                'cfg': _cfg()}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
if a.adabn:   # unsupervised test-time BN recalibration on target-batch images from TRAINING FOVs (split 0; labels never read)
    assert a.protocol in ('b1to0', 'b0to1'), 'adabn: batch hold-out protocols only'
    tb = 0 if a.protocol == 'b1to0' else 1; tgt = np.random.default_rng(0).permutation(np.where((b == tb) & (sp == 0) & has)[0])[:a.adabn]
    for m_ in model['enc'].modules():
        if isinstance(m_, nn.modules.batchnorm._BatchNorm): m_.reset_running_stats(); m_.momentum = None
    model.train()
    with torch.no_grad():
        for raw, ph, g, *_ in mk(tgt, False): fwd(prep(raw, ph, g, False))
    ra = evaluate(save=True, tag='_adabn'); print('ADABN', json.dumps(ra, default=float), flush=True)
    json.dump({'args': vars(a), 'n_adabn': len(tgt), 'final': ra}, open(a.out + '/result_adabn.json', 'w'), indent=1, default=float)
if run: run.finish()
json.dump({'args': vars(a), 'final': log[-1], 'log': log, 'n_train': len(tr), 'n_eval': len(ev), 'targets': names}, open(a.out + '/result.json', 'w'), indent=1, default=float)
