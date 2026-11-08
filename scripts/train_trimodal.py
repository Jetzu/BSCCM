"""Trimodal v0 (ImageBind-style, label-free = pivot). Shared space for Giemsa (G), label-free LED (L), markers (M).
Pairs used: G-L (histology cells, unstained) and L-M (antibody-stained cells). G-M is never paired -> emergent.
G = frozen DinoBloom features (precomputed) + MLP adapter; L = CNN on 23 LED channels; M = MLP on masked marker vector.
Losses: SigLIP(L,G) + SigLIP(L,M) + masked marker regression + gate-class CE, both heads on the SHARED normalized space
so they can be applied to any modality (G->M "emergent" prediction = head(z_G)).
usage: train_trimodal.py --out DIR [--steps 6000] [--gfeat emb/giemsa_dinobloom_B.npy] [--channels all]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from bsccm.data import LEDMarkerSet, chan_list, W
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--gfeat', default=W + '/emb/giemsa_dinobloom_B.npy'); ap.add_argument('--channels', default='all'); ap.add_argument('--arch', default='resnet18')
ap.add_argument('--dim', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--workers', type=int, default=14)
ap.add_argument('--w-reg', type=float, default=1.0); ap.add_argument('--w-cls', type=float, default=0.5); ap.add_argument('--w-gl', type=float, default=1.0); ap.add_argument('--w-lm', type=float, default=1.0)
ap.add_argument('--eval-every', type=int, default=1000); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
T = np.load(W + '/targets.npz'); K = len(T['names']); NC = int(T['cls'].max()) + 1; names = list(T['names'])
sp, hrow, Mk, Y, cls, batch = T['split'], T['hist_row'], T['mask'], T['Y'], T['cls'], T['batch']
GF = torch.tensor(np.load(a.gfeat), dtype=torch.float32); GF = (GF - GF.mean(0)) / (GF.std(0) + 1e-6)
stained = Mk.any(1)
gl_tr = np.where((sp == 0) & (hrow >= 0))[0]; gl_va = np.where((sp == 1) & (hrow >= 0))[0]; gl_te = np.where((sp == 2) & (hrow >= 0))[0]
lm_tr = np.where((sp == 0) & (stained | (cls >= 0)))[0]; lm_te = np.where((sp == 2) & (stained | (cls >= 0)))[0]
ym = np.array([Y[lm_tr, j][Mk[lm_tr, j]].mean() for j in range(K)], np.float32); ys = np.array([Y[lm_tr, j][Mk[lm_tr, j]].std() + 1e-6 for j in range(K)], np.float32)
ym_t, ys_t = torch.tensor(ym).cuda(), torch.tensor(ys).cuda()
def loader(rows, train, bs=a.bs, nw=a.workers // 2):
    return torch.utils.data.DataLoader(LEDMarkerSet(rows, a.channels, train=train), batch_size=bs, shuffle=train, drop_last=train,
                                       num_workers=nw, pin_memory=True, persistent_workers=train, prefetch_factor=4 if nw else None)
def forever(dl):
    while True:
        for b in dl: yield b
class SigLIP(nn.Module):
    def __init__(s): super().__init__(); s.t = nn.Parameter(torch.tensor(np.log(10.))); s.b = nn.Parameter(torch.tensor(-10.))
    def forward(s, za, zb):
        lg = za @ zb.T * s.t.exp() + s.b; lab = 2 * torch.eye(len(za), device=za.device) - 1
        return -F.logsigmoid(lab * lg).sum() / len(za)
class Tri(nn.Module):
    def __init__(s):
        super().__init__(); D = a.dim
        s.L = timm.create_model(a.arch, pretrained=False, num_classes=0, in_chans=len(chan_list(a.channels))); s.pL = nn.Linear(s.L.num_features, D)
        s.G = nn.Sequential(nn.Linear(GF.shape[1], 1024), nn.GELU(), nn.Dropout(0.2), nn.LayerNorm(1024), nn.Linear(1024, D))
        s.M = nn.Sequential(nn.Linear(2 * K, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, D))
        s.hM = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, K)); s.hC = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, NC))
        s.sGL, s.sLM = SigLIP(), SigLIP()
    def zL(s, x): return F.normalize(s.pL(s.L(x)).float(), dim=-1)
    def zG(s, f): return F.normalize(s.G(f), dim=-1)
    def zM(s, y, m): return F.normalize(s.M(torch.cat([((y - ym_t) / ys_t) * m, m.float()], 1)), dim=-1)
    def heads(s, z): z = z * np.sqrt(a.dim); return s.hM(z), s.hC(z)   # heads act on the shared (unit-norm, rescaled) space
net = Tri().cuda(); net.L = net.L.to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
it_gl, it_lm = forever(loader(gl_tr, True)), forever(loader(lm_tr, True))
cl = lambda x: x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
@torch.no_grad()
def embed_L(rows):
    net.eval(); Z = []
    for x, *_ in loader(rows, False, 512, 8):
        with torch.autocast('cuda', torch.bfloat16): Z.append(net.zL(cl(x)))
    net.train(); return torch.cat(Z)
def retrieval(za, zb):   # row i of za matches row i of zb
    s = za @ zb.T; rk = (s > s.diag()[:, None]).sum(1)
    return {'R@1': float((rk < 1).float().mean()), 'R@5': float((rk < 5).float().mean()), 'R@10': float((rk < 10).float().mean()), 'medr': float(rk.float().median() + 1)}
def pearson(p, y): p, y = p - p.mean(), y - y.mean(); return float((p * y).sum() / (p.norm() * y.norm() + 1e-9))
@torch.no_grad()
def evaluate(rows_gl, rows_lm, tag):
    net.eval(); r = {}
    zl = embed_L(rows_gl); zg = net.zG(GF[hrow[rows_gl]].cuda())
    for k, v in retrieval(zg, zl).items(): r[f'{tag}/G->L_{k}'] = v
    for k, v in retrieval(zl, zg).items(): r[f'{tag}/L->G_{k}'] = v
    r[f'{tag}/n_gl'] = len(rows_gl); r[f'{tag}/chance_R@1'] = 1 / len(rows_gl)
    # emergent G->M: shared heads applied to z_G vs to z_L of the SAME unstained cells (consistency, not ground truth)
    pg, cg = net.heads(zg); pl, clg = net.heads(zl)
    r[f'{tag}/GM_vs_LM_consistency_pearson_mean'] = float(np.mean([pearson(pg[:, j], pl[:, j]) for j in range(K)]))
    r[f'{tag}/G_vs_L_class_agreement'] = float((cg.argmax(1) == clg.argmax(1)).float().mean())
    if rows_lm is not None:
        rl = rows_lm[np.random.default_rng(0).permutation(len(rows_lm))[:20000]]
        zl2 = embed_L(rl); p, c = net.heads(zl2); p = (p * ys_t + ym_t).cpu(); y = torch.tensor(Y[rl]); m = torch.tensor(Mk[rl])
        pr = {n: pearson(p[m[:, j], j], y[m[:, j], j]) for j, n in enumerate(names) if m[:, j].sum() > 20}
        r.update({f'{tag}/L->M_pearson/{n}': v for n, v in pr.items()}); r[f'{tag}/L->M_pearson_mean'] = float(np.mean(list(pr.values())))
        lab = torch.tensor(cls[rl]) >= 0
        if lab.any(): r[f'{tag}/L_cls_acc'] = float((c.argmax(1).cpu()[lab] == torch.tensor(cls[rl])[lab]).float().mean())
        full = m[:, :4].all(1)   # 'all' cells: 4 group targets observed
        if full.sum() > 50:
            idx = torch.where(full)[0][:2000]; zm = net.zM(y[idx].cuda(), m[idx].cuda())
            for k, v in retrieval(zm, zl2[idx]).items(): r[f'{tag}/M->L_{k}(n={len(idx)})'] = v
    net.train(); return r
try:
    import wandb; run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), name=os.path.basename(a.out.rstrip('/')), id=os.path.basename(a.out.rstrip('/')), resume='allow', config=vars(a))
except Exception as e: print('wandb off:', e); run = None
ck = a.out + '/ckpt.pt'; step0 = 0; log = []; best = -1
if os.path.exists(ck):
    s = torch.load(ck, map_location='cuda', weights_only=False); net.load_state_dict(s['m']); opt.load_state_dict(s['o']); sched.load_state_dict(s['s']); step0, log, best = s['step'], s['log'], s['best']; print('resumed', step0, flush=True)
t0 = time.time(); acc = {}
for step in range(step0, a.steps):
    xg, _, _, _, gg = next(it_gl); xm, y, m, c, _ = next(it_lm)
    y, m, c = y.cuda(non_blocking=True), m.cuda(non_blocking=True), c.cuda(non_blocking=True)
    with torch.autocast('cuda', torch.bfloat16):
        zl = net.zL(cl(torch.cat([xg, xm])))
    zl_g, zl_m = zl[:len(xg)], zl[len(xg):]
    zg = net.zG(GF[hrow[gg.numpy()]].cuda(non_blocking=True))
    st = m.any(1); zm = net.zM(y[st], m[st])
    l_gl = net.sGL(zg, zl_g); l_lm = net.sLM(zm, zl_m[st])
    pm, pc = net.heads(zl_m)
    reg = (((pm - (y - ym_t) / ys_t) ** 2) * m).sum(0) / m.sum(0).clamp(min=1); reg = reg[m.any(0)].mean()
    ce = F.cross_entropy(pc, c, ignore_index=-1) if (c >= 0).any() else pc.sum() * 0
    loss = a.w_gl * l_gl + a.w_lm * l_lm + a.w_reg * reg + a.w_cls * ce
    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
    for k, v in (('loss', loss), ('sig_GL', l_gl), ('sig_LM', l_lm), ('reg', reg), ('ce', ce)): acc[k] = acc.get(k, 0) + float(v)
    if (step + 1) % 100 == 0:
        msg = {f'train/{k}': v / 100 for k, v in acc.items()}; acc = {}
        print(f'step {step+1}/{a.steps} ' + ' '.join(f'{k}={v:.4f}' for k, v in msg.items()) + f' {time.time()-t0:.0f}s', flush=True)
        if run: run.log(msg, step=step + 1)
    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        r = evaluate(gl_va, None, 'val'); r['step'] = step + 1; log.append(r); print(json.dumps(r), flush=True)
        if run: run.log(r, step=step + 1)
        score = r['val/G->L_R@10'] + r['val/L->G_R@10']
        sd = {'m': net.state_dict(), 'o': opt.state_dict(), 's': sched.state_dict(), 'step': step + 1, 'log': log, 'best': max(best, score)}
        torch.save(sd, ck + '.tmp'); os.replace(ck + '.tmp', ck)
        if score > best: best = score; torch.save(net.state_dict(), a.out + '/best.pt.tmp'); os.replace(a.out + '/best.pt.tmp', a.out + '/best.pt')
# final test with best checkpoint
net.load_state_dict(torch.load(a.out + '/best.pt', map_location='cuda'))
rt = evaluate(gl_te, lm_te, 'test'); print(json.dumps(rt), flush=True)
if run: run.log(rt, step=a.steps); run.finish()
json.dump({'args': vars(a), 'test': rt, 'val_log': log}, open(a.out + '/result.json', 'w'), indent=1)
