"""N011: image-level GS encoder bound to the FROZEN LF pivot (fixes the cross-slide failure of frozen DinoBloom features, N009).
GS encoder = DinoBloom-B (ViT-B/14, 224 px); first --freeze blocks frozen, rest + projection trained. Strong stain/optics augmentation
on GPU (Wright RGB is isotropic -> rotations/flips allowed): random resized crop, rot90/flip, per-channel optical-density (stain) scaling,
hue/saturation jitter, gamma, Gaussian blur, noise, background tint.
LF side: frozen features (lf_feats_all.npy of the chosen LF encoder) -> trainable projection P_L.
Teacher: ridge LF-features -> 13 targets fitted on stained TRAINING cells (closed form), applied to GS cells' own LF = distillation target.
Loss: SigLIP(z_GS, z_LF) + w_distill * masked MSE(student head(z_GS), teacher(LF)).   autofl truth on GS cells is EVAL ONLY.
Eval on held-out stained slide (batch 1): retrieval GS<->LF, autofl truth r (student vs teacher vs truth), student-vs-teacher r per marker.
usage: train_gs_bind.py --lf-feats DIR --out DIR [--steps 6000] [--freeze 8] [--w-distill 1.0] [--holdout batch1|fov]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
ap = argparse.ArgumentParser()
ap.add_argument('--lf-feats', required=True); ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--freeze', type=int, default=8)
ap.add_argument('--dim', type=int, default=256); ap.add_argument('--w-distill', type=float, default=1.0); ap.add_argument('--holdout', default='batch1')
ap.add_argument('--aug', type=int, default=1); ap.add_argument('--stain-norm', default='none', choices=['none', 'macenko']); ap.add_argument('--dann', type=float, default=0.0); ap.add_argument('--target-zip', default=''); ap.add_argument('--anchor-ce', type=float, default=0.0); ap.add_argument('--target-half', default='A'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--workers', type=int, default=12)
ap.add_argument('--targets', default='targets_v4.npz')   # teacher-ridge labels + eval truth; v5 = N023 label QC
ap.add_argument('--backbone', default='B', choices=['B', 'L'])   # N027: DinoBloom-B (ViT-B/14, 768) or DinoBloom-L (ViT-L/14, 1024)
ap.add_argument('--aux-feats', default=''); ap.add_argument('--aux-targets', default=''); ap.add_argument('--aux-names', default='autofl')   # N025 two-teacher: these outputs come from a second LF teacher
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/' + a.targets); names = list(T['names']); K = len(names)
Y, Mk, sp, b, hrow = T['Y'], T['mask'], T['split'], T['batch'], T['hist_row']
gidx = np.where(hrow >= 0)[0]; gidx = gidx[np.argsort(hrow[gidx])]                      # histology row r <-> global gidx[r]
if a.holdout == 'batch1': tr_r, te_r = np.where(b[gidx] == 0)[0], np.where(b[gidx] == 1)[0]
elif a.holdout == 'batch0': tr_r, te_r = np.where(b[gidx] == 1)[0], np.where(b[gidx] == 0)[0]
else: tr_r, te_r = np.where(sp[gidx] == 0)[0], np.where(sp[gidx] == 2)[0]
LF = torch.tensor(np.load(os.path.join(a.lf_feats, 'lf_feats_all.npy')), dtype=torch.float32)
stained_tr = np.where((sp == 0) & Mk[:, :12].any(1) & (hrow < 0))[0]
mu, sd = LF[stained_tr].mean(0), LF[stained_tr].std(0) + 1e-6; LF = (LF - mu) / sd
# --- teacher: closed-form ridge per target on stained training cells (GPU), applied to GS cells' own LF features
def ridge_fit(X, y, alpha=10.0):
    Xb = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1); A = Xb.T @ Xb + alpha * torch.eye(Xb.shape[1], device=X.device); A[-1, -1] -= alpha
    return torch.linalg.solve(A, Xb.T @ y)
Wt = torch.zeros(LF.shape[1] + 1, K, device=dev)
for j in range(K):
    r = np.where((sp == 0) & Mk[:, j] & (hrow < 0))[0]; r = np.random.default_rng(j).permutation(r)[:60000]
    Wt[:, j] = ridge_fit(LF[r].to(dev), torch.tensor(Y[r, j], device=dev))
LFg = LF[gidx].to(dev); teach = torch.cat([LFg, torch.ones(len(LFg), 1, device=dev)], 1) @ Wt      # (Nh, K) teacher predictions on GS cells
AUX = []
if a.aux_feats:   # two-teacher distillation: listed targets are re-fitted on a second teacher's LF features with a second label file
    TA = np.load(W + '/' + a.aux_targets); YA, MA = TA['Y'], TA['mask']; namesA = [str(n) for n in TA['names']]
    for key in ('hist_row', 'split', 'batch', 'slide'):   # red-team N025 F5: rows of both target files must be the same cells
        assert np.array_equal(TA[key], T[key]), f'aux targets {a.aux_targets}: {key} differs from {a.targets}'
    LFA = torch.tensor(np.load(os.path.join(a.aux_feats, 'lf_feats_all.npy')), dtype=torch.float32); assert LFA.shape[0] == LF.shape[0], 'aux feature rows differ'
    LFA = (LFA - LFA[stained_tr].mean(0)) / (LFA[stained_tr].std(0) + 1e-6); LFAg = LFA[gidx].to(dev)
    for nm in a.aux_names.split(','):
        j = names.index(nm); jA = namesA.index(nm); AUX.append(j)   # j indexes the student/teacher outputs, jA the aux label file (F5)
        r = np.where((sp == 0) & MA[:, jA] & (hrow < 0))[0]; r = np.random.default_rng(j).permutation(r)[:60000]
        teach[:, j] = torch.cat([LFAg, torch.ones(len(LFAg), 1, device=dev)], 1) @ ridge_fit(LFA[r].to(dev), torch.tensor(YA[r, jA], device=dev))
    del LFA
    print(f'aux teacher for {a.aux_names} from {a.aux_feats} ({a.aux_targets})', flush=True)
H = np.load(W + '/pack/histology.npy', mmap_mode='r'); scale = float(np.percentile(H[:500:5], 99.9))
class GSSet(torch.utils.data.Dataset):
    def __init__(s, rows): s.rows = rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        r = int(s.rows[i]); return torch.from_numpy(np.array(H[r]).astype(np.int16)), r      # uint16 <= 4095 -> int16 view-safe
class ExtSet(torch.utils.data.Dataset):   # unlabelled external smear images (labels never read here)
    def __init__(s, zpath, half, anchors=None):
        import zipfile; s.zp = zpath; s.z = None
        names = sorted(n for n in zipfile.ZipFile(zpath).namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n and not os.path.basename(n).startswith('.'))
        s.names = names[0::2] if half == 'A' else names[1::2]
        if anchors: s.names = [n for n in s.names if n.split('/')[-2] in anchors]     # anchor classes only (eos/baso never seen)
        s.cls = [anchors.index(n.split('/')[-2]) if anchors and n.split('/')[-2] in anchors else -1 for n in s.names]
    def __len__(s): return len(s.names)
    def __getitem__(s, i):
        import zipfile, io
        from PIL import Image
        if s.z is None: s.z = zipfile.ZipFile(s.zp)
        im = Image.open(io.BytesIO(s.z.read(s.names[i]))).convert('RGB').resize((398, 398), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im, np.uint8).copy()), s.cls[i]
def loader(rows, train):
    return torch.utils.data.DataLoader(GSSet(rows), batch_size=a.bs, shuffle=train, drop_last=train, num_workers=a.workers, pin_memory=True, persistent_workers=train, prefetch_factor=4)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
def _macenko_stains(od, beta=0.15):
    # od: (3, P) optical density of one image; returns 3x2 stain matrix (unit columns) and 99th-pct concentrations
    m = od.sum(0) > beta; o = od[:, m] if m.sum() > 200 else od
    c = torch.cov(o); ev, V = torch.linalg.eigh(c); V = V[:, 1:]                       # top-2 eigenvectors (3x2)
    pr = V.T @ o; ang = torch.atan2(pr[1], pr[0]); lo, hi = torch.quantile(ang, 0.01), torch.quantile(ang, 0.99)
    v1 = V @ torch.stack([torch.cos(lo), torch.sin(lo)]); v2 = V @ torch.stack([torch.cos(hi), torch.sin(hi)])
    v1, v2 = v1 * torch.sign(v1.sum()), v2 * torch.sign(v2.sum())
    S = torch.stack([v1, v2], 1)
    S = S / S.norm(dim=0, keepdim=True); C = torch.linalg.lstsq(S, od).solution
    return S, torch.quantile(C[:, m] if m.sum() > 200 else C, 0.99, dim=1)
MAC_REF = None
def macenko(x):   # x (B,3,H,W) in [0,1] -> stain-normalised to MAC_REF
    B, _, Hh, Ww = x.shape; out = torch.empty_like(x)
    for i in range(B):
        od = -torch.log(x[i].reshape(3, -1).clamp(min=1e-3)); S, cmax = _macenko_stains(od)
        # order the two stains consistently with the reference (match by cosine)
        if (S[:, 0] @ MAC_REF[0][:, 0]) < (S[:, 1] @ MAC_REF[0][:, 0]): S, cmax = S[:, [1, 0]], cmax[[1, 0]]
        C = torch.linalg.lstsq(S, od).solution * (MAC_REF[1] / cmax.clamp(min=1e-3))[:, None]
        out[i] = torch.exp(-(MAC_REF[0] @ C)).reshape(3, Hh, Ww)
    return out.clamp(0, 1)
def prep(x, train):
    x = (x.to(dev).float() / scale).clamp(0, 1).permute(0, 3, 1, 2)                      # (B,3,398,398) in [0,1]
    return pipeline(x, train)
def pipeline(x, train):   # x (B,3,398,398) float in [0,1] (BSCCM or resized external smears)
    if a.stain_norm == 'macenko': x = macenko(x)
    B = len(x)
    if train and a.aug:
        od = -torch.log(x.clamp(min=1e-3))                                                  # optical density
        od = od * torch.exp(0.25 * torch.randn(B, 3, 1, 1, device=dev))                     # per-channel stain intensity
        od = od * torch.exp(0.15 * torch.randn(B, 1, 1, 1, device=dev))                     # overall stain amount
        x = torch.exp(-od)
        x = x * (1 + 0.04 * torch.randn(B, 3, 1, 1, device=dev))                            # background tint / white balance
        g = torch.exp(0.2 * torch.randn(B, 1, 1, 1, device=dev)); x = x.clamp(1e-4, 1) ** g    # gamma
        sig = 2.0 * torch.rand(B, device=dev); k = torch.arange(-6, 7, device=dev).float()
        ker = torch.exp(-0.5 * (k[None] / sig.clamp(min=0.05)[:, None]) ** 2); ker = ker / ker.sum(1, keepdim=True)
        kk = ker[:, None, :].expand(B, 3, 13).reshape(B * 3, 1, 1, 13)
        y = F.conv2d(F.pad(x.reshape(1, B * 3, 398, 398), (6, 6, 0, 0), mode='reflect'), kk, groups=B * 3)
        x = F.conv2d(F.pad(y, (0, 0, 6, 6), mode='reflect'), kk.transpose(2, 3), groups=B * 3).reshape(B, 3, 398, 398)
        x = x + 0.02 * torch.randn_like(x)
        # smooth illumination / stain-thickness gradient across the crop (seen as slide-position artefact, N012)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 398, device=dev), torch.linspace(-1, 1, 398, device=dev), indexing='ij')
        gvec = 0.08 * torch.randn(B, 2, device=dev); x = x * (1 + gvec[:, 0, None, None, None] * xx + gvec[:, 1, None, None, None] * yy)
        # axial chromatic aberration: extra blur on ONE random colour channel (sequential R/G/B LED focus differences)
        c = int(torch.randint(3, (1,))); sg = float(1.5 * torch.rand(1)); 
        if sg > 0.2:
            kk1 = torch.exp(-0.5 * (k / sg) ** 2); kk1 = (kk1 / kk1.sum()).view(1, 1, 1, 13)
            xc = x[:, c:c + 1]; xc = F.conv2d(F.pad(xc, (6, 6, 0, 0), mode='reflect'), kk1); xc = F.conv2d(F.pad(xc, (0, 0, 6, 6), mode='reflect'), kk1.transpose(2, 3))
            x = torch.cat([x[:, :c], xc, x[:, c + 1:]], 1)
        s = (0.8 + 0.2 * torch.rand(B, device=dev)); t = (torch.rand(B, 2, device=dev) - 0.5) * 0.2 # resized crop: scale 0.8-1, shift
        th = torch.zeros(B, 2, 3, device=dev); th[:, 0, 0] = s; th[:, 1, 1] = s; th[:, :, 2] = t
        x = F.grid_sample(x, F.affine_grid(th, (B, 3, 224, 224), align_corners=False), padding_mode='reflection', align_corners=False)
        k90 = int(torch.randint(4, (1,))); x = torch.rot90(x, k90, (2, 3))
        if torch.rand(1) < 0.5: x = x.flip(3)
    else:
        x = F.interpolate(x, size=224, mode='bilinear', antialias=True)
    return (x.clamp(0, 1) - MEAN) / STD
if a.stain_norm == 'macenko':
    Ss, Cs = [], []
    for r in np.random.default_rng(0).choice(len(gidx), 400, replace=False):
        od = -torch.log((torch.tensor(np.asarray(H[r], np.float32), device=dev) / scale).clamp(1e-3, 1).permute(2, 0, 1).reshape(3, -1))
        S_, c_ = _macenko_stains(od); Ss.append(S_); Cs.append(c_)
    S0 = torch.stack(Ss)                                                                  # align stain order to the first image, then median
    for i in range(len(Ss)):
        if (S0[i][:, 0] @ S0[0][:, 0]) < (S0[i][:, 1] @ S0[0][:, 0]): S0[i] = S0[i][:, [1, 0]]; Cs[i] = Cs[i][[1, 0]]
    Sref = S0.median(0).values; Sref = Sref / Sref.norm(dim=0, keepdim=True); MAC_REF = (Sref, torch.stack(Cs).median(0).values)
    print('macenko reference stains', Sref.cpu().numpy().round(3).tolist(), 'cmax', MAC_REF[1].cpu().numpy().round(3).tolist(), flush=True)
vit = timm.create_model({'B': 'vit_base_patch14_dinov2', 'L': 'vit_large_patch14_dinov2'}[a.backbone], pretrained=False, num_classes=0, img_size=224)
sd_ = torch.load(f'{W}/models/DinoBloom-{a.backbone}.pth', map_location='cpu', weights_only=True)['teacher']; VD = vit.num_features
vit.load_state_dict({k[9:]: v for k, v in sd_.items() if k.startswith('backbone.') and k != 'backbone.mask_token'}, strict=True)
for p in vit.patch_embed.parameters(): p.requires_grad = False
vit.cls_token.requires_grad = False; vit.pos_embed.requires_grad = False
for i, blk in enumerate(vit.blocks):
    for p in blk.parameters(): p.requires_grad = i >= a.freeze
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, o))
net = nn.ModuleDict({'vit': vit, 'PG': mlp(VD, a.dim), 'PL': mlp(LF.shape[1], a.dim), 'S': mlp(a.dim, K), 'D': mlp(VD, 1), 'C': mlp(a.dim, 3)}).to(dev)
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None
tsig = nn.Parameter(torch.tensor(math.log(10.), device=dev)); bsig = nn.Parameter(torch.tensor(-10., device=dev))
params = [p for p in net.parameters() if p.requires_grad] + [tsig, bsig]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
def encode(x, feats=False):
    with torch.autocast('cuda', torch.bfloat16): h = net['vit'](x).float()
    z = F.normalize(net['PG'](h), dim=-1); return (z, h) if feats else z
zL = lambda r: F.normalize(net['PL'](LFg[torch.as_tensor(r, device=dev)]), dim=-1)
def pearson(p, y): p, y = p - p.mean(), y - y.mean(); return float((p * y).sum() / (p.norm() * y.norm() + 1e-9))
def retr(A, B):
    s = A @ B.T; rk = ((s >= s.diag()[:, None]).sum(1) - 1).float()
    return {'R@1': float((rk < 1).float().mean()), 'R@10': float((rk < 10).float().mean()), 'medr': float(rk.median() + 1)}
@torch.no_grad()
def evaluate():
    net.eval(); Z, R = [], []
    for x, r in loader(te_r, False): Z.append(encode(prep(x, False))); R.append(r)
    zg = torch.cat(Z); rr = torch.cat(R).numpy(); zl = zL(rr); out = {f'GS->LF_{k}': v for k, v in retr(zg, zl).items()}
    out.update({f'LF->GS_{k}': v for k, v in retr(zl, zg).items()}); out['n_test'] = len(rr); out['chance_R@1'] = 1 / len(rr)
    stud = net['S'](zg * math.sqrt(a.dim)).cpu(); tch = teach[torch.as_tensor(rr, device=dev)].cpu()
    j = names.index('autofl'); gg = gidx[rr]; v = Mk[gg, j]; y = torch.tensor(Y[gg, j])
    out['autofl_truth_r/student_GS'] = pearson(stud[v, j], y[v]); out['autofl_truth_r/teacher_LF'] = pearson(tch[v, j], y[v])
    out['student_vs_teacher_r_mean12'] = float(np.mean([pearson(stud[:, k], tch[:, k]) for k in range(12)]))
    out.update({f'student_vs_teacher_r/{names[k]}': pearson(stud[:, k], tch[:, k]) for k in range(K)})
    # error-bound ingredients (N016-validated): b = RMSE(student vs teacher) on the held-out GS slide; a = RMSE(teacher vs truth) on
    # held-out STAINED cells (split 2, non-GS) of the same batch as the test slide; bounds on the unmeasurable GS->MK MSE
    tb = int(b[gidx[rr[0]]]); inb = torch.as_tensor(b[gidx[rr]] == tb)   # F7: student-teacher error b on the SAME batch as teacher error a
    for k in range(12):
        bk = float(((stud[inb, k] - tch[inb, k]) ** 2).mean().sqrt()); ev = np.where((sp == 2) & Mk[:, k] & (hrow < 0) & (b == tb))[0]
        if len(ev) < 50 or k in AUX: continue   # bound ingredient a uses the primary teacher only
        tp = (torch.cat([LF[ev].to(dev), torch.ones(len(ev), 1, device=dev)], 1) @ Wt[:, k]).cpu(); yt = torch.tensor(Y[ev, k])
        ak = float(((tp - yt) ** 2).mean().sqrt()); vy = float(yt.var())
        out[f'bound/{names[k]}'] = dict(a=ak, b=bk, teacher_r=pearson(tp, yt), mse_lo=(ak - bk) ** 2, mse_hi=(ak + bk) ** 2, mse_cia=ak ** 2 + bk ** 2,
                                       var_y=vy, r2_cia=1 - (ak ** 2 + bk ** 2) / vy, r2_worst=1 - (ak + bk) ** 2 / vy, r2_best=1 - (ak - bk) ** 2 / vy)
    np.savez(a.out + '/test_preds.npz', rows=rr, gidx=gidx[rr], student=stud.numpy(), teacher=tch.numpy(), z=zg.cpu().numpy())
    net.train(); return out
log = []; t0 = time.time(); it = iter(loader(tr_r, True)); acc = {}
ANCH = ['lymphocyte', 'neutrophil', 'monocyte']
if a.anchor_ce > 0:
    _lfc = np.load('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N008-external-lineage-anchor-pbc-acevedo-mgg/gs_lf_cls3_pred.npz')['pred']  # 0 lym,1 gran,2 mono
    GS_CLS = torch.tensor(_lfc, device=dev)                                    # same order as ANCH (gran ~ neutrophil-dominated)
    _anc = ExtSet(a.target_zip, a.target_half, ANCH); print(f'anchor CE: {len(_anc)} external anchor images {ANCH}', flush=True)
    _anc_loader = lambda: iter(torch.utils.data.DataLoader(_anc, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True))
    it_a = _anc_loader()
if a.dann > 0 and a.target_zip:
    _ext = ExtSet(a.target_zip, a.target_half); print(f'DANN target: {len(_ext)} unlabelled external images (half {a.target_half})', flush=True)
    _ext_loader = lambda: iter(torch.utils.data.DataLoader(_ext, batch_size=a.bs, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=True))
    it_t = _ext_loader()
else:
    it_t = iter(loader(te_r, True)) if a.dann > 0 else None   # UNLABELLED images of the held-out slide (no markers/autofl used)
for step in range(a.steps):
    try: x, r = next(it)
    except StopIteration: it = iter(loader(tr_r, True)); x, r = next(it)
    zg, hg = encode(prep(x, True), feats=True); zl = zL(r.numpy())
    l_dann = torch.zeros((), device=dev)
    if a.dann > 0:
        try: xt, _ = next(it_t)
        except StopIteration: it_t = _ext_loader() if a.target_zip else iter(loader(te_r, True)); xt, _ = next(it_t)
        xt01 = (xt.to(dev).float() / 255.).permute(0, 3, 1, 2) if a.target_zip else None
        _, ht = encode(pipeline(xt01, True) if a.target_zip else prep(xt, True), feats=True); hh = torch.cat([hg, ht]); dl = torch.cat([torch.zeros(len(hg)), torch.ones(len(ht))]).to(dev)
        l_dann = F.binary_cross_entropy_with_logits(net['D'](_GRL.apply(hh, 1.0))[:, 0], dl)
    lg = zg @ zl.T * tsig.exp() + bsig; lab = 2 * torch.eye(len(zg), device=dev) - 1; l_sig = -F.logsigmoid(lab * lg).sum() / len(zg)
    m = torch.ones(len(r), K, device=dev)      # distil ALL 13 teacher outputs (incl. the teacher's autofl PREDICTION); measured autofl truth stays eval-only
    dist = (((net['S'](zg * math.sqrt(a.dim)) - teach[r.to(dev)]) ** 2) * m).sum() / m.sum()
    l_anc = torch.zeros((), device=dev)
    if a.anchor_ce > 0:
        try: xa, ya = next(it_a)
        except StopIteration: it_a = _anc_loader(); xa, ya = next(it_a)
        za = encode(pipeline((xa.to(dev).float() / 255.).permute(0, 3, 1, 2), True))
        l_anc = F.cross_entropy(net['C'](za * math.sqrt(a.dim)), ya.to(dev)) + F.cross_entropy(net['C'](zg * math.sqrt(a.dim)), GS_CLS[r.to(dev)])
    loss = l_sig + a.w_distill * dist + a.dann * l_dann + a.anchor_ce * l_anc; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    acc['anc'] = acc.get('anc', 0) + float(l_anc); acc['dann'] = acc.get('dann', 0) + float(l_dann); acc['sig'] = acc.get('sig', 0) + float(l_sig); acc['dist'] = acc.get('dist', 0) + float(dist)
    if (step + 1) % 200 == 0: print(f"step {step+1} sig {acc['sig']/200:.3f} dist {acc['dist']/200:.4f} dann {acc['dann']/200:.3f} anc {acc['anc']/200:.3f} {time.time()-t0:.0f}s", flush=True); acc = {}
    if (step + 1) % 2000 == 0 or step + 1 == a.steps:
        e = evaluate(); e['step'] = step + 1; log.append(e); print(json.dumps(e, default=float), flush=True)
torch.save({k: v for k, v in net.state_dict().items()}, a.out + '/gs_bind.pt')
json.dump({'args': vars(a), 'final': log[-1], 'log': log}, open(a.out + '/result.json', 'w'), indent=1, default=float)
