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
ap.add_argument('--enc-lr-mult', type=float, default=1.0)   # N053: encoder lr = lr x this (pretrained ViTs: 0.1); heads keep lr
ap.add_argument('--drop-path', type=float, default=0.1)   # N053: stochastic depth of the pretrained ViT backbones (bsccm.encoders only)
# N060--N062: all new recipe options are inert/omitted from cfg at default.
ap.add_argument('--head', choices=['mse', 'hlgauss', 'twopart'], default='mse')
ap.add_argument('--hl-mse', type=float, default=0.1)
ap.add_argument('--gate-w', type=float, default=0.5)
ap.add_argument('--pos-q', type=float, default=0.98)
ap.add_argument('--fov-re', type=float, default=0.0)
ap.add_argument('--re-shrink', type=float, default=1.0)
ap.add_argument('--pair-loss', type=float, default=0.0)
ap.add_argument('--fov-batch', type=int, default=0)
ap.add_argument('--w-autofl', type=float, default=1.0)
ap.add_argument('--pearson-w', type=float, default=0.0)   # N060 (lit_noisy_ceiling): + w * mean over present targets of (1 - batch Pearson r)
ap.add_argument('--diag-aug', type=float, default=0.0)   # N068: p of the anti-diagonal reflection + LED swap (DF_xx_Bottom<->Right, DPC B<->R, T<->L; n066_d4_aniso)
ap.add_argument('--diag-tta', type=int, default=0)   # N068: eval prediction = mean over identity + anti-diagonal view
ap.add_argument('--drop-led', default='')   # N068: comma-separated LED names marked ABSENT (data 0, SSL presence plane 0) in every sample, e.g. LED119
ap.add_argument('--censor-loss', type=int, default=0)   # N073: clip-floor labels (targets 'censored') are upper bounds: no loss when prediction < label
# N074: optional training-only fluorescence decoder; defaults omitted from legacy cfg.
ap.add_argument('--dense-aux', default='')
ap.add_argument('--dense-w', type=float, default=0.5)
ap.add_argument('--dense-width', type=int, default=64)
ap.add_argument('--aux-label', default='')   # N075: targets npz (same rows/columns) of an ALTERNATIVE label version -> separate auxiliary head; eval uses the main head
ap.add_argument('--aux-w', type=float, default=1.0)
ap.add_argument('--init-full', default='')   # N076: ckpt.pt of a trained run -> load the FULL model state (encoder + heads) before training
ap.add_argument('--train-panel', default='')   # N076: comma-separated antibody conditions (bsccm_index 'antibodies'), e.g. all -> train only on those slides' cells
ap.add_argument('--panel-head', action='store_true')
ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
if not np.isfinite(a.phys_aug) or not 0 <= a.phys_aug <= 1: ap.error('--phys-aug must be in [0,1]')
if not np.isfinite(a.consist) or a.consist < 0: ap.error('--consist must be nonnegative')
if a.consist > 0 and a.phys_aug == 0: ap.error('--consist requires --phys-aug > 0')
for _key in ('hl_mse', 'gate_w', 'fov_re', 're_shrink', 'pair_loss', 'w_autofl'):
    if not np.isfinite(getattr(a, _key)) or getattr(a, _key) < 0: ap.error(f'--{_key.replace("_", "-")} must be finite and nonnegative')
if not np.isfinite(a.pos_q) or not 0 < a.pos_q < 1: ap.error('--pos-q must be in (0,1)')
if a.fov_batch < 0: ap.error('--fov-batch must be nonnegative')
if a.panel_head and a.head != 'mse': ap.error('--panel-head requires --head mse')
if a.head == 'hlgauss' and a.fov_re: ap.error('HL-Gauss and FOV readout adapters are separate arms; combine with pair loss instead')
NNEW = a.head != 'mse' or a.panel_head or a.fov_re > 0 or a.pair_loss > 0 or a.fov_batch > 0 or a.w_autofl != 1 or a.pearson_w > 0
if NNEW:
    if a.protocol not in ('b1to0', 'b0to1'): ap.error('N060--N062 require a single training draw')
    if a.include_gs or a.dann > 0 or a.adabn: ap.error('N060--N062 source-only requires --include-gs 0 --dann 0 --adabn 0')
DENSE = bool(a.dense_aux)
if DENSE:
    if a.protocol not in ('b1to0', 'b0to1'): ap.error('--dense-aux requires a single training draw: b1to0 or b0to1')
    if a.include_gs or a.dann > 0 or a.adabn: ap.error('--dense-aux source-only requires --include-gs 0 --dann 0 --adabn 0')
    if not np.isfinite(a.dense_w) or a.dense_w < 0: ap.error('--dense-w must be finite and nonnegative')
    if a.dense_width < 1: ap.error('--dense-width must be positive')
    if a.diag_aug > 0: ap.error('--dense-aux with --diag-aug: fluorescence crops are offset from LF by a per-channel instrument shift (N074 red-team MAJOR-1, reproduced) -> reflected views would get contradictory targets')
    if not (a.arch.startswith('resnet') or a.arch == 'cytoland'):
        ap.error('--dense-aux requires a supported NCHW pre-pool spatial map (resnet family or cytoland); token-only dinov3/dinobloom unsupported')
    from bsccm.dense_aux import DenseDecoder, DenseTargets, pooled_with_spatial, roll_batch, align_target, dense_mse
DEVICE = a.device
PHYS = a.phys_aug > 0
if PHYS:
    if a.protocol not in ('b1to0', 'b0to1'): ap.error('N041 requires a single source draw: --protocol b1to0 or b0to1')
    if a.include_gs or a.dann > 0 or a.adabn: ap.error('N041 source-only requires --include-gs 0 --dann 0 --adabn 0')
T = np.load(os.path.join(W, a.targets)); names = list(T['names']); K = len(names)
Y, Mk, c3, c10, sp, b = T['Y'], T['mask'], T['cls3'], T['cls10'], T['split'], T['batch']
has = Mk.any(1) | (c3 >= 0)
if not a.include_gs: has &= T['hist_row'] < 0   # keep the 15,712 GS cells (label: autofl only) out of LF training -> clean GS autofl test
if a.protocol == 'fov': tr = (sp == 0) & has; ev = (sp == 2) & has
elif a.protocol == 'b1to0': tr = (b == 1) & has; ev = (b == 0) & (sp == 2) & has
elif a.protocol == 'b0to1': tr = (b == 0) & has; ev = (b == 1) & (sp == 2) & has
else: raise SystemExit('protocol?')
tr, ev = np.where(tr)[0], np.where(ev)[0]
if a.train_panel:   # N076: multi-marker fine-tune = training-draw cells of the given antibody conditions only (evaluation rows unchanged)
    import pandas as _pd; _ab = _pd.read_parquet(os.path.join(W, 'bsccm_index.parquet'), columns=['antibodies']).antibodies.to_numpy()
    tr = tr[np.isin(_ab[tr], a.train_panel.split(','))]; print(f'N076 train panel {a.train_panel}: {len(tr)} training rows', flush=True)
if a.max_train: tr = np.random.default_rng(0).permutation(tr)[:a.max_train]
if DENSE:
    dense_targets = DenseTargets(a.dense_aux, tr, b, sp, os.path.join(W, 'bsccm_index.parquet'))
print(f'train {len(tr)} eval {len(ev)} protocol {a.protocol}', flush=True)
if len(tr) < a.bs: ap.error('--max-train / source rows must supply at least one full batch')
from bsccm.heads import HLGauss, TwoPart, PanelHead, TrainAdapters, FOVBatchSampler, masked_mean, pair_difference_loss
SETUP = None
if NNEW:
    import hashlib
    if K != 13: ap.error('N060--N062 require 13 targets')
    # Only tr enters statistics, fingerprints, adapter tables, or FOV sampling.
    _source = hashlib.sha256()
    for _arr in (tr, Y[tr], Mk[tr], T['slide'][tr], T['fov'][tr]):
        _source.update(np.ascontiguousarray(_arr).tobytes())
    SETUP = {'version': 1, 'source_sha256': _source.hexdigest(),
             'counts': Mk[tr].sum(0).tolist(), 'names': [str(n) for n in names]}
    _, _fi = np.unique(np.stack([T['slide'][tr], T['fov'][tr]], 1), axis=0, return_inverse=True)
    FOV_INDEX = torch.full((len(Y),), -1, dtype=torch.long, device=DEVICE)
    FOV_INDEX[torch.as_tensor(tr, device=DEVICE)] = torch.as_tensor(_fi, device=DEVICE)
    TARGET_WEIGHT = torch.ones(K, device=DEVICE); TARGET_WEIGHT[names.index('autofl')] = a.w_autofl
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
if a.censor_loss:   # N073: carried as the first extra batch element (pz[0]); --pseudo also uses pz -> not combinable
    if a.pseudo: ap.error('--censor-loss is not combinable with --pseudo')
    LAB['C'] = T['censored'].astype(bool)
AUX = bool(a.aux_label)
if AUX:   # N075 label separation (Wei et al. 2023): appended after 'C' -> read as pz[-2], pz[-1]
    if a.pseudo: ap.error('--aux-label is not combinable with --pseudo')
    _T2 = np.load(os.path.join(W, a.aux_label)); assert [str(n) for n in _T2['names']] == [str(n) for n in names] and len(_T2['Y']) == len(Y), 'aux label layout'
    _Y2 = _T2['Y'].astype(np.float32); _M2 = _T2['mask'].astype(bool) & np.isfinite(_Y2) & Mk; _Y2 = np.where(_M2, _Y2, 0).astype(np.float32)
    LAB['Y2'] = _Y2; LAB['M2'] = _M2; print(f'N075 aux label {a.aux_label}: {int(_M2[tr].sum())} training cell x target entries', flush=True)
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
def mk(rows, train):
    kw = dict(num_workers=a.workers, pin_memory=DEVICE == 'cuda')
    if a.workers: kw.update(persistent_workers=True, prefetch_factor=8)
    if train and a.fov_batch:
        sampler = FOVBatchSampler(T['fov'][rows], T['slide'][rows], a.bs, a.fov_batch, a.seed)
        return torch.utils.data.DataLoader(RawSet(rows, LAB), batch_sampler=sampler, **kw)
    return torch.utils.data.DataLoader(RawSet(rows, LAB), batch_size=a.bs, shuffle=train, drop_last=train, **kw)
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
    phys_aug = PhysAug(a.phys_aug, phys_calibration['gain_adu_per_e'], phys_calibration['offset_sd_adu'], device=DEVICE,
                       gain_log_sd=phys_calibration['gain_log_sd'])
    print('N041 source-border calibration:', {k: phys_calibration[k] for k in ('source_draw', 'gain_adu_per_e', 'offset_sd_adu', 'gain_log_sd', 'rows_sha256')}, flush=True)
gnorm = GPUNorm(DEVICE); CH = chan_list(a.channels)
DIAG = None
if a.diag_aug > 0 or a.diag_tta:   # N068: the only exact symmetry of the acquired LED set (full rings are not sums of halves; no Top/Left halves)
    from bsccm.data import LED as _LED
    _sw = {f'DF_{k}_Bottom': f'DF_{k}_Right' for k in (50, 60, 70, 80)}; _sw.update({'DPC_Bottom': 'DPC_Right', 'DPC_Top': 'DPC_Left'}); _sw.update({v: k for k, v in list(_sw.items())})
    _nm = [_LED[c] for c in CH]
    if 'LED119' in _nm and 'LED119' not in a.drop_led.split(','): ap.error('--diag-aug/--diag-tta: LED119 has no anti-diagonal partner; drop it (--drop-led LED119 or --channels no119)')
    if any(_sw.get(n, n) not in _nm for n in _nm): ap.error('--diag-aug/--diag-tta: channel set not closed under the Bottom<->Right swap')
    DIAG = torch.tensor([_nm.index(_sw.get(n, n)) for n in _nm] + ([len(_nm)] if a.phase else []), device=DEVICE)
DROP = None
if a.drop_led:   # N068: absent channels = zero data + zero presence (the SSL encoders were pretrained with random channel dropout this way)
    from bsccm.data import LED as _LED2
    DROP = torch.tensor([CH.index(_LED2.index(n)) for n in a.drop_led.split(',')], device=DEVICE)
def diag(x):   # anti-diagonal reflection x'[i,j] = x[W-1-j, H-1-i] (= np.rot90(x, 2).T), then channel k <- partner(k)
    return x.flip(-1).flip(-2).transpose(-1, -2)[:, DIAG]
_cnt = np.bincount(c10[tr][c10[tr] >= 0], minlength=10).astype(np.float32); C10W = torch.tensor(1 / np.sqrt(np.maximum(_cnt, 1)), device=DEVICE); C10W = C10W / C10W[_cnt > 0].mean()
def prep(raw, ph, g, train, dense_target=None):   # raw int16 records -> normalised LF on GPU (+-4 px translation when training)
    raw = raw.to(DEVICE, non_blocking=True)
    if train and phys_aug is not None: raw = phys_aug(raw)   # raw counts BEFORE flat-field/slide/log normalisation; phase untouched
    if dense_target is None:
        x = gnorm(raw, ph.to(DEVICE, non_blocking=True), g.to(DEVICE, non_blocking=True), CH, bool(a.phase), shift=4 if train else 0)
    else:
        if not train or not DENSE: raise AssertionError('dense target only permitted in source training prep')
        x = gnorm(raw, ph.to(DEVICE, non_blocking=True), g.to(DEVICE, non_blocking=True), CH, bool(a.phase), shift=0)
        # GPUNorm samples dy/dx on CPU and torch.roll AFTER normalisation. Reproduce exactly.
        shifts = torch.randint(-4, 5, (2, len(x))).T
        x = roll_batch(x, shifts)
        dense_sel = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    if a.lowpass > 0:   # depthwise separable Gaussian, reflect padding; applied before zoom
        r_ = int(np.ceil(3 * a.lowpass)); k_ = torch.exp(-0.5 * (torch.arange(-r_, r_ + 1, device=x.device, dtype=x.dtype) / a.lowpass) ** 2); k_ = k_ / k_.sum()
        c_ = x.shape[1]; x = F.pad(x, (r_, r_, r_, r_), mode='reflect')
        x = F.conv2d(x, k_.view(1, 1, 1, -1).repeat(c_, 1, 1, 1), groups=c_); x = F.conv2d(x, k_.view(1, 1, -1, 1).repeat(c_, 1, 1, 1), groups=c_)
    if a.zoom > 1:
        s_ = 128 // a.zoom; o = (128 - s_) // 2; x = F.interpolate(x[..., o:o + s_, o:o + s_], size=(128, 128), mode='bilinear', align_corners=False)
    if train and a.diag_aug > 0:
        sel = torch.rand(len(x), device=x.device) < a.diag_aug
        if sel.any(): x[sel] = diag(x[sel])
        if dense_target is not None: dense_sel = sel
    if dense_target is not None:
        target, weights = align_target(dense_target, shifts, dense_sel, a.zoom)
        return x, target, weights
    return x
dtr, dev = mk(tr, True), mk(ev, False)
if a.dann > 0:
    assert a.protocol in ('b1to0', 'b0to1'), 'dann: batch hold-out protocols only'
    tb = 0 if a.protocol == 'b1to0' else 1; tgt = np.where((b == tb) & (sp == 0))[0]      # unlabelled target images, training FOVs only
    def _forever(dl):
        while True:
            for z in dl: yield z
    it_tgt = _forever(mk(tgt, True)); print(f'DANN: {len(tgt)} unlabelled target-batch images (split 0 only)', flush=True)
SSL = bool(a.init_ssl)
from bsccm.encoders import ARCHS as _N053, LFBackbone
if a.arch in _N053:   # N053: pretrained ViT/ConvNeXt with a learned per-LED gate and a 24-channel patch embedding
    if SSL or a.stem != 'std' or a.adabn: ap.error('N053 archs: no --init-ssl / --stem / --adabn')
    net = LFBackbone(a.arch, nin, a.drop_path)
else: net = timm.create_model(a.arch, pretrained=False, num_classes=0, in_chans=48 if SSL else nin)
if SSL:
    assert a.channels == 'all' and a.phase == 1, 'SSL init expects all 23 LED + phase'
    net.load_state_dict(torch.load(a.init_ssl, map_location='cpu', weights_only=True), strict=True); print('init from SSL', a.init_ssl, flush=True)
if a.stem == 'nomaxpool': net.maxpool = nn.Identity()   # stride 2 instead of 4 before layer1 (weights unchanged)
NF = net.num_features * (2 if a.dual_zoom > 1 else 1)
model = nn.ModuleDict({'enc': net, 'reg': nn.Linear(NF, K), 'c3': nn.Linear(NF, 3), 'c10': nn.Linear(NF, 10),
                      'dom': nn.Sequential(nn.Linear(NF, 256), nn.ReLU(), nn.Linear(256, 1))}).to(DEVICE)
if DENSE: model['dense'] = DenseDecoder(net.num_features, a.dense_width).to(DEVICE)
if AUX: model['reg2'] = nn.Linear(NF, K).to(DEVICE)   # N075: alternative-label head (training only)
if a.init_full:   # N076: continue a trained model (same architecture/heads); strict so a recipe mismatch fails loudly
    model.load_state_dict(torch.load(a.init_full, map_location='cpu', weights_only=False)['m'], strict=True); print('init FULL from', a.init_full, flush=True)
if a.head == 'hlgauss': model['reg'] = HLGauss(NF, Y[tr], Mk[tr]).to(DEVICE)
elif a.head == 'twopart': model['reg'] = TwoPart(NF, names, Y[tr], Mk[tr], a.pos_q).to(DEVICE)
elif a.panel_head: model['reg'] = PanelHead(NF, names).to(DEVICE)
if a.fov_re or a.panel_head:
    model['readout'] = TrainAdapters(T['fov'][tr], T['slide'][tr], K, bool(a.fov_re), a.panel_head).to(DEVICE)
    SLIDE_INDEX = torch.full((len(Y),), -1, dtype=torch.long, device=DEVICE)
    SLIDE_INDEX[torch.as_tensor(tr, device=DEVICE)] = torch.as_tensor(model['readout'].si, device=DEVICE)
    assert np.array_equal(model['readout'].fi, _fi)
if SETUP is not None:
    for _key in ('edges', 'thresholds', 'class_weight', 'gate_active', 'design'):
        if hasattr(model['reg'], _key): SETUP[_key] = getattr(model['reg'], _key).detach().cpu().tolist()
    print('N060--N062 source setup:', json.dumps({k: v for k, v in SETUP.items() if k != 'edges'}), flush=True)
steps = a.epochs * len(dtr)
if a.enc_lr_mult == 1.0:   # legacy single param group (older checkpoints resume / dump unchanged)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
else:   # N053: lower lr for the pretrained encoder
    _enc = list(model['enc'].parameters()); _ids = {id(q) for q in _enc}; _rest = [q for q in model.parameters() if id(q) not in _ids]
    opt = torch.optim.AdamW([{'params': _enc, 'lr': a.lr * a.enc_lr_mult}, {'params': _rest, 'lr': a.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr * a.enc_lr_mult, a.lr], total_steps=steps, pct_start=0.1)
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None
def fwd(x, dense_features=None):
    pres = torch.ones_like(x)
    if DROP is not None: x = x.index_fill(1, DROP, 0.0); pres = pres.index_fill(1, DROP, 0.0)
    if SSL: x = torch.cat([x, pres], 1)
    with torch.autocast(DEVICE, torch.bfloat16, enabled=DEVICE == 'cuda'):
        if dense_features is None:
            h = model['enc'](x.contiguous(memory_format=torch.channels_last))
        else:
            h, spatial = pooled_with_spatial(model['enc'], x.contiguous(memory_format=torch.channels_last))
            dense_features.append(spatial)   # original full/zoomed input view only; never an evaluation call
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
        xx = prep(raw, ph, g, False); p, l3, l10, h = fwd(xx)
        if a.diag_tta: p2, l32, l102, _ = fwd(diag(xx)); p = (p + p2) / 2; l3 = (l3 + l32) / 2; l10 = (l10 + l102) / 2
        P.append(p.cpu().numpy()); C3.append(l3.argmax(1).cpu().numpy()); C10.append(l10.argmax(1).cpu().numpy()); G.append(g.numpy())
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
def _run_args():
    args = vars(a).copy()
    if not DENSE:
        for key in ('dense_aux', 'dense_w', 'dense_width'): args.pop(key)
    if not AUX:
        for key in ('aux_label', 'aux_w'): args.pop(key)
    for key in ('init_full', 'train_panel'):
        if not getattr(a, key): args.pop(key)
    return args
try:
    if a.dump_train_preds: raise RuntimeError('no wandb for a teacher dump (would append to the teacher run)')
    import wandb; run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), name=os.path.basename(a.out.rstrip('/')), id=os.path.basename(a.out.rstrip('/')), resume='allow', config=_run_args())
except Exception as e: print('wandb off:', e); run = None
_NEW_DEFAULTS = {'stem': 'std', 'zoom': 1, 'dual_zoom': 1, 'lowpass': 0.0, 'phys_aug': 0.0, 'consist': 0.0, 'pseudo': '', 'pseudo_w': 0.3, 'enc_lr_mult': 1.0, 'drop_path': 0.1}   # options added after older checkpoints were written: omitted at default so they still resume
_NEW_DEFAULTS.update(head='mse', hl_mse=0.1, gate_w=0.5, pos_q=0.98, fov_re=0.0,
                     re_shrink=1.0, pair_loss=0.0, fov_batch=0, w_autofl=1.0, panel_head=False, device='cuda', pearson_w=0.0, diag_aug=0.0, diag_tta=0, drop_led='', censor_loss=0)
_NEW_DEFAULTS.update(dense_aux='', dense_w=0.5, dense_width=64)
_NEW_DEFAULTS.update(aux_label='', aux_w=1.0)
_NEW_DEFAULTS.update(init_full='', train_panel='')
def _cfg():
    cfg = {k: v for k, v in _run_args().items() if k not in ('workers', 'out', 'adabn', 'dump_train_preds') and not (k in _NEW_DEFAULTS and v == _NEW_DEFAULTS[k])}
    if DENSE: cfg['dense_setup'] = dense_targets.identity
    if PHYS: cfg['phys_calibration'] = phys_calibration   # pin source row IDs, fitted gain/offsets and method on resume
    if PSEUDO_ID is not None: cfg['pseudo_id'] = PSEUDO_ID   # N040 F3
    if SETUP is not None: cfg['n060_062_setup'] = SETUP
    return cfg
ck = a.out + '/ckpt.pt'; ep0 = 0; log = []
if os.path.exists(ck):
    s = torch.load(ck, map_location=DEVICE, weights_only=False)
    cfg = _cfg()
    if s.get('cfg') != cfg or not np.array_equal(s.get('tr'), tr) or not np.array_equal(s.get('ev'), ev):
        raise SystemExit(f'refusing to resume {ck}: config/rows differ from the checkpoint (red-team F3)')
    model.load_state_dict(s['m']); opt.load_state_dict(s['o']); sched.load_state_dict(s['s']); ep0, log = s['ep'], s['log']; print('resumed', ep0, flush=True)
if SETUP is not None and not a.dump_train_preds:
    with open(a.out + '/head_setup.json', 'w') as f: json.dump(SETUP, f, indent=2)
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
    tl = n = 0; dl = 0.0
    if a.fov_batch: dtr.batch_sampler.epoch = ep
    for raw, ph, g, y, m, k3, k10, wt, *pz in dtr:
        if a.consist > 0:   # transfer once; both prep calls independently sample raw-count nuisances and the existing translation
            raw, ph, g = raw.to(DEVICE, non_blocking=True), ph.to(DEVICE, non_blocking=True), g.to(DEVICE, non_blocking=True)
        if DENSE:
            dense_target, dense_panel = dense_targets.get(g, DEVICE)   # asserts tr membership AND source draw before reading pixels
            xx, dense_target, dense_weights = prep(raw, ph, g, True, dense_target)
            dense_features = []
            _last_h = fwd(xx, dense_features)
            dense_loss = dense_mse(model['dense'](dense_features[0], dense_panel), dense_target, dense_weights)
            del dense_features
            y, m, k3, k10, wt = y.to(DEVICE), m.to(DEVICE), k3.to(DEVICE), k10.to(DEVICE), wt.to(DEVICE); p, l3, l10, _ = _last_h
        else:
            y, m, k3, k10, wt = y.to(DEVICE), m.to(DEVICE), k3.to(DEVICE), k10.to(DEVICE), wt.to(DEVICE); _last_h = fwd(prep(raw, ph, g, True)); p, l3, l10, _ = _last_h
        if a.censor_loss:   # true value <= label on censored cells: replace the target by the (detached) prediction where it is already below
            cz = pz[0].to(DEVICE); y = torch.where(cz & (p.detach() < y), p.detach().to(y.dtype), y)
        if not NNEW:  # legacy arithmetic and model state layout remain unchanged at defaults
            reg = (((p - y) ** 2) * m * wt).sum(0) / (m * wt).sum(0).clamp(min=1e-6); reg = reg[m.any(0)].mean()
        else:
            gg = g.to(DEVICE); fit_p = p
            if 'readout' in model: fit_p = model['readout'](p, l3, FOV_INDEX[gg], SLIDE_INDEX[gg])
            mse = (fit_p - y).square()
            value = model['reg'].cell_loss(_last_h[3], y) + a.hl_mse * mse if a.head == 'hlgauss' else mse
            reg = masked_mean(value, m, wt, TARGET_WEIGHT)
            if a.head == 'twopart': reg = reg + a.gate_w * model['reg'].gate_loss(_last_h[3], y, m, wt, TARGET_WEIGHT)
            if 'readout' in model: reg = reg + model['readout'].penalty(a.fov_re, a.re_shrink, TARGET_WEIGHT)
            if a.pair_loss: reg = reg + a.pair_loss * pair_difference_loss(p, y, m, FOV_INDEX[gg], TARGET_WEIGHT, gg)
            if a.pearson_w > 0:   # per-target masked batch Pearson on the shared prediction (label units), targets weighted like the MSE
                mm = m.float() * wt; nn_ = mm.sum(0); ok = (nn_ >= 8) & (TARGET_WEIGHT > 0)
                pc = p - (p * mm).sum(0) / nn_.clamp(min=1); yc = y - (y * mm).sum(0) / nn_.clamp(min=1)
                rr = (pc * yc * mm).sum(0) / torch.sqrt((pc.square() * mm).sum(0) * (yc.square() * mm).sum(0) + 1e-6)   # eps inside sqrt: finite grad at zero variance
                if ok.any(): reg = reg + a.pearson_w * (1 - rr[ok]).mean()
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
            hh = torch.cat([hs, ht]); dl_ = torch.cat([torch.zeros(len(hs)), torch.ones(len(ht))]).to(DEVICE)
            ce = ce + a.dann * F.binary_cross_entropy_with_logits(model['dom'](_GRL.apply(hh, 1.0))[:, 0], dl_)
        consistency = p.new_zeros(())
        if a.consist > 0:
            p2, _, _, _ = fwd(prep(raw, ph, g, True))
            consistency = F.mse_loss(p, p2)   # all K regression predictions; symmetric gradients, no labels or target-draw images
        loss = reg + a.w_cls * ce
        if DENSE: loss = loss + a.dense_w * dense_loss
        if AUX:   # N075: masked MSE of the alternative-label head on the shared features
            y2, m2 = pz[-2].to(DEVICE), pz[-1].to(DEVICE).float(); p2a = model['reg2'](_last_h[3])
            r2 = ((p2a - y2) ** 2 * m2).sum(0) / m2.sum(0).clamp(min=1e-6); loss = loss + a.aux_w * (r2[m2.any(0)].mean() if m2.any() else p2a.sum() * 0)
        if a.pseudo:
            py, pm = pz[0].to(DEVICE), pz[1].to(DEVICE).float()
            preg = (((p.float() - py) ** 2) * pm).sum(0) / pm.sum(0).clamp(min=1e-6); loss = loss + a.pseudo_w * (preg[pm.any(0)].mean() if pm.any() else p.sum() * 0)
        if a.consist > 0: loss = loss + a.consist * consistency
        opt.zero_grad(set_to_none=True); loss.backward()
        if DENSE:   # N074 red-team MINOR-2: clip decoder separately so the scalar path keeps the control's effective step size
            nn.utils.clip_grad_norm_([q for k_, q in model.named_parameters() if not k_.startswith('dense.')], 2.0); nn.utils.clip_grad_norm_(model['dense'].parameters(), 2.0)
            dl += float(dense_loss.detach())
        else: nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        tl += loss.item(); n += 1; step += 1
        if step % 200 == 0:
            print(f'ep {ep} step {step}/{steps} loss {tl/n:.4f} {time.time()-t0:.0f}s', flush=True)
            if run: run.log({'train/loss': tl / n}, step=step)
    r = evaluate(save=(ep + 1 == a.epochs)); r.update(epoch=ep + 1, train_loss=tl / n, secs=time.time() - t0)
    if DENSE: r['dense_loss'] = dl / max(n, 1)
    log.append(r); print(json.dumps(r, default=float), flush=True)
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
    json.dump({'args': _run_args(), 'n_adabn': len(tgt), 'final': ra}, open(a.out + '/result_adabn.json', 'w'), indent=1, default=float)
if run: run.finish()
_extra = {'led_weights': model['enc'].led_weights(), 'led_channels': CH + (['phase'] if a.phase else [])} if a.arch in _N053 else {}   # N053
json.dump({'args': _run_args(), 'final': log[-1], 'log': log, 'n_train': len(tr), 'n_eval': len(ev), 'targets': names, **_extra}, open(a.out + '/result.json', 'w'), indent=1, default=float)
