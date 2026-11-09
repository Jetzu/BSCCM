"""Stage 2 (N009): bind GS and MK to the FROZEN LF pivot (ImageBind-style), on precomputed features.
Inputs: --lf-feats  <dir>/lf_feats_all.npy (N,d) f16/f32 from the chosen frozen LF encoder (row = global index)
        --gs-feats  emb/giemsa_dinobloom_B.npy (15,712 x 768, hist_row order); targets_v3.npz.
Modules (all small): P_L (LF proj), E_G (GS adapter MLP), E_M (marker SET encoder: tokens = marker-id embedding + value MLP
[z, censored]; 2-layer transformer; mean pool), heads on the shared unit sphere: H_M (13 targets), H_C (cls3).
Losses: SigLIP(GS,LF) on GS-LF pairs; SigLIP(MK,LF) where pairs (i!=j) that observed the same marker with |z_i-z_j| < tau are IGNORED
(not pushed apart: near-identical marker readouts are not negatives); masked MSE of H_M(z_L); CE of H_C(z_L);
optional distillation: S_G (GS student head) regresses the LF-head prediction on the paired LF cell (--w-distill).
Eval (held-out stained slide = batch 1 GS cells; FOV test split for LF-MK):
  retrieval GS<->LF; MK->LF on 'all' cells; LF->MK r per target; and on the REAL GS cells, autofluorescence (measured on them) via
  (i) head transfer H_M(z_G), (ii) distilled student S_G(z_G), (iii) teacher H_M(z_L of the same cell) vs truth.
usage: train_bind.py --lf-feats DIR --out DIR [--steps 20000] [--w-distill 1.0]"""
import argparse, json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap = argparse.ArgumentParser()
ap.add_argument('--lf-feats', required=True); ap.add_argument('--gs-feats', default='/hnvme/workspace/v181bb10-bsccm/emb/giemsa_dinobloom_B.npy')
ap.add_argument('--out', required=True); ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=1024)
ap.add_argument('--dim', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--tau', type=float, default=0.25)
ap.add_argument('--w-gl', type=float, default=1.0); ap.add_argument('--w-lm', type=float, default=1.0); ap.add_argument('--w-reg', type=float, default=1.0)
ap.add_argument('--w-cls', type=float, default=0.5); ap.add_argument('--w-distill', type=float, default=1.0); ap.add_argument('--gs-holdout', default='batch1')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--per-slide-norm', type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = 'cuda'
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v3.npz')
Y, Mk, Cz, c3, sp, b, hrow, names = T['Y'], T['mask'], T['censored'], T['cls3'], T['split'], T['batch'], T['hist_row'], list(T['names']); K = len(names)
LF = torch.tensor(np.load(os.path.join(a.lf_feats, 'lf_feats_all.npy')), dtype=torch.float32)
GSf = torch.tensor(np.load(a.gs_feats), dtype=torch.float32)
Yt, Mt, Ct = torch.tensor(Y), torch.tensor(Mk), torch.tensor(Cz)
gs_cells = np.where(hrow >= 0)[0]
if a.gs_holdout == 'batch1': gs_tr, gs_te = gs_cells[b[gs_cells] == 0], gs_cells[b[gs_cells] == 1]
else: gs_tr, gs_te = gs_cells[sp[gs_cells] == 0], gs_cells[sp[gs_cells] == 2]
mk_tr = np.where((sp == 0) & Mk[:, :12].any(1))[0]; mk_te = np.where((sp == 2) & Mk[:, :12].any(1))[0]
lab_tr = np.where((sp == 0) & (c3 >= 0))[0]
_lf_fit = np.unique(np.concatenate([gs_tr, mk_tr, lab_tr]))                 # red-team F6: fit standardisation on TRAINING rows only
LF = (LF - LF[_lf_fit].mean(0)) / (LF[_lf_fit].std(0) + 1e-6)
_g = torch.tensor(hrow[gs_tr]); GSf = (GSf - GSf[_g].mean(0)) / (GSf[_g].std(0) + 1e-6)
if a.per_slide_norm:   # domain-specific normalisation: each slide's features standardised by its OWN unlabelled statistics
    sl = torch.tensor(T['slide'])                          # (transductive on unlabelled features of the test slide; no labels used)
    for v in sl.unique():
        r = torch.where(sl == v)[0]; LF[r] = (LF[r] - LF[r].mean(0)) / (LF[r].std(0) + 1e-6)
    _gi = np.where(hrow >= 0)[0]; _gi = _gi[np.argsort(hrow[_gi])]; gsl = sl[torch.tensor(_gi)]   # slide of each histology row (row r <-> global _gi[r])
    for v in gsl.unique():
        r = torch.where(gsl == v)[0]; GSf[r] = (GSf[r] - GSf[r].mean(0)) / (GSf[r].std(0) + 1e-6)
class SetEnc(nn.Module):
    def __init__(s, D):
        super().__init__(); s.id = nn.Embedding(K, 128); s.val = nn.Sequential(nn.Linear(2, 128), nn.GELU(), nn.Linear(128, 128))
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True, norm_first=True), 2); s.out = nn.Linear(128, D)
    def forward(s, y, m, cz):   # y,m,cz: (B,K) ; only observed markers become tokens (padding mask for the rest)
        tok = s.id.weight[None] + s.val(torch.stack([y, cz.float()], -1)); h = s.tf(tok, src_key_padding_mask=~m)
        h = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1); return s.out(h)
class SigLIP(nn.Module):
    def __init__(s): super().__init__(); s.t = nn.Parameter(torch.tensor(np.log(10.))); s.b = nn.Parameter(torch.tensor(-10.))
    def forward(s, za, zb, ignore=None):
        lg = za @ zb.T * s.t.exp() + s.b; lab = 2 * torch.eye(len(za), device=za.device) - 1
        l = -F.logsigmoid(lab * lg)
        if ignore is not None: l = l * (~ignore).float()
        return l.sum() / len(za)
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 1024), nn.GELU(), nn.Dropout(0.1), nn.LayerNorm(1024), nn.Linear(1024, o))
net = nn.ModuleDict({'PL': mlp(LF.shape[1], a.dim), 'EG': mlp(GSf.shape[1], a.dim), 'EM': SetEnc(a.dim), 'HM': mlp(a.dim, K), 'HC': mlp(a.dim, 3),
                     'SG': mlp(a.dim, K), 'DG': mlp(GSf.shape[1], 1), 'sGL': SigLIP(), 'sLM': SigLIP()}).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
zn = lambda x: F.normalize(x, dim=-1); sc = np.sqrt(a.dim)
zL = lambda i: zn(net['PL'](LF[i].to(dev))); zG = lambda i: zn(net['EG'](GSf[hrow[i]].to(dev)))
zM = lambda i: zn(net['EM'](Yt[i].to(dev), Mt[i].to(dev), Ct[i].to(dev)))
def near_dup(i):   # ignore-matrix for SigLIP(MK,LF): IDENTICAL observed surface-marker set (12 targets; autofl excluded) and
    m = Mt[i, :12].to(dev).float(); y = Yt[i, :12].to(dev)   # every observed value within tau -> indistinguishable readouts, not negatives
    same = (m[:, None] == m[None]).all(-1) & (m.sum(1)[:, None] > 0)
    diff = ((y[:, None] - y[None]).abs() * m[:, None]).amax(-1)
    ign = same & (diff < a.tau); ign.fill_diagonal_(False); return ign
def pearson(p, y): p, y = p - p.mean(), y - y.mean(); return float((p * y).sum() / ((p.norm() * y.norm()) + 1e-9))
def retr(A, B):
    s = A @ B.T; rk = ((s >= s.diag()[:, None]).sum(1) - 1).float(); return {'R@1': float((rk < 1).float().mean()), 'R@10': float((rk < 10).float().mean()), 'medr': float(rk.median() + 1)}
@torch.no_grad()
def evaluate():
    net.eval(); r = {}
    zg, zl = zG(gs_te), zL(gs_te)
    r.update({f'GS->LF_{k}': v for k, v in retr(zg, zl).items()}); r.update({f'LF->GS_{k}': v for k, v in retr(zl, zg).items()}); r['gs_test_n'] = len(gs_te)
    # autofluorescence on REAL GS cells (measured on them): truth vs teacher / head transfer / distilled student
    j = names.index('autofl'); v = Mk[gs_te, j]; y = torch.tensor(Y[gs_te, j][v])
    teach = net['HM'](zl * sc)[:, j].cpu()[v]; head = net['HM'](zg * sc)[:, j].cpu()[v]; stud = net['SG'](zg * sc)[:, j].cpu()[v]
    r['autofl_GS_truth_r/teacher_LF'] = pearson(teach, y); r['autofl_GS_truth_r/head_transfer_GS'] = pearson(head, y); r['autofl_GS_truth_r/distilled_GS'] = pearson(stud, y)
    r['autofl_GS_n'] = int(v.sum())
    direct = net['DG'](GSf[hrow[gs_te]].to(dev))[:, 0].cpu()[v]; r['autofl_GS_truth_r/DIRECT_supervised_GS_ceiling'] = pearson(direct, y)
    # marker predictions consistency on GS cells (no truth): student/head vs teacher
    pt, ph, ps = net['HM'](zl * sc).cpu(), net['HM'](zg * sc).cpu(), net['SG'](zg * sc).cpu()
    r['GSvsTeacher_r_mean/head_transfer'] = float(np.mean([pearson(ph[:, k], pt[:, k]) for k in range(12)]))
    r['GSvsTeacher_r_mean/distilled'] = float(np.mean([pearson(ps[:, k], pt[:, k]) for k in range(12)]))
    r['b2_rmse/distilled'] = float(((ps[:, :12] - pt[:, :12]) ** 2).mean().sqrt())
    # LF->MK on held-out stained cells
    te = np.random.default_rng(0).permutation(mk_te)[:30000]; p = net['HM'](zL(te) * sc).cpu()
    rs = {}
    for k, n in enumerate(names[:12]):
        vv = Mk[te, k]
        if vv.sum() > 30: rs[n] = pearson(p[vv, k], torch.tensor(Y[te, k][vv]))
    r.update({f'LF->MK_r/{n}': x for n, x in rs.items()}); r['LF->MK_r_mean'] = float(np.mean(list(rs.values())))
    r['a2_rmse/teacher'] = float(np.sqrt(np.mean([((p[Mk[te, k], k] - torch.tensor(Y[te, k][Mk[te, k]])) ** 2).mean().item() for k in range(12) if Mk[te, k].sum() > 30])))
    full = te[Mk[te, 8:12].all(1)][:2000]
    if len(full) > 100: r.update({f'MK->LF_{k}(all-cells)': v for k, v in retr(zM(full), zL(full)).items()})
    net.train(); return r
log = []; t0 = time.time(); acc = {}
for step in range(a.steps):
    ig = torch.tensor(np.random.choice(gs_tr, min(a.bs, len(gs_tr)), replace=False)); im = torch.tensor(np.random.choice(mk_tr, a.bs, replace=False))
    ic = torch.tensor(np.random.choice(lab_tr, 256, replace=False))
    zg, zlg = zG(ig), zL(ig); zm, zlm = zM(im), zL(im); zlc = zL(ic)
    l_gl = net['sGL'](zg, zlg); l_lm = net['sLM'](zm, zlm, near_dup(im))
    pm = net['HM'](zlm * sc); m = Mt[im].to(dev).float(); y = Yt[im].to(dev)
    reg = (((pm - y) ** 2) * m).sum(0) / m.sum(0).clamp(min=1); reg = reg[m.sum(0) > 0].mean()
    ce = F.cross_entropy(net['HC'](zlc * sc), torch.tensor(c3[ic.numpy()], dtype=torch.long, device=dev))
    with torch.no_grad(): teacher = net['HM'](zlg * sc)            # LF teacher prediction on the paired LF cell (stop-grad)
    dist = F.mse_loss(net['SG'](zg * sc), teacher)
    jA = names.index('autofl'); vg = torch.tensor(Mk[ig.numpy(), jA], device=dev)   # direct GS->autofl ceiling (separate head on raw GS features, no gradient into the shared space)
    dg = F.mse_loss(net['DG'](GSf[hrow[ig.numpy()]].to(dev))[:, 0][vg], Yt[ig, jA].to(dev)[vg]) if vg.any() else torch.zeros((), device=dev)
    loss = dg + a.w_gl * l_gl + a.w_lm * l_lm + a.w_reg * reg + a.w_cls * ce + a.w_distill * dist
    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
    for k, v in (('loss', loss), ('gl', l_gl), ('lm', l_lm), ('reg', reg), ('ce', ce), ('distill', dist)): acc[k] = acc.get(k, 0) + float(v)
    if (step + 1) % 500 == 0:
        print(f'step {step+1} ' + ' '.join(f'{k}={v/500:.4f}' for k, v in acc.items()) + f' {time.time()-t0:.0f}s', flush=True); acc = {}
    if (step + 1) % 5000 == 0 or step + 1 == a.steps:
        r = evaluate(); r['step'] = step + 1; log.append(r); print(json.dumps(r), flush=True)
torch.save(net.state_dict(), a.out + '/bind.pt')
json.dump({'args': vars(a), 'final': log[-1], 'log': log}, open(a.out + '/result.json', 'w'), indent=1)
