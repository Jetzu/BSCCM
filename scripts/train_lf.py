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
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
T = np.load(W + '/targets_v3.npz'); names = list(T['names']); K = len(names)
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
LAB = {'Y': Y, 'M': Mk, 'c3': c3.astype(np.int64), 'c10': c10.astype(np.int64)}
mk = lambda rows, train: torch.utils.data.DataLoader(RawSet(rows, LAB), batch_size=a.bs, shuffle=train, drop_last=train,
                                                      num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=8)
gnorm = GPUNorm('cuda'); CH = chan_list(a.channels)
def prep(raw, ph, g, train):   # raw int16 records -> normalised LF on GPU (+-4 px translation when training)
    return gnorm(raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True), CH, bool(a.phase), shift=4 if train else 0)
dtr, dev = mk(tr, True), mk(ev, False)
net = timm.create_model(a.arch, pretrained=False, num_classes=0, in_chans=nin)
model = nn.ModuleDict({'enc': net, 'reg': nn.Linear(net.num_features, K), 'c3': nn.Linear(net.num_features, 3), 'c10': nn.Linear(net.num_features, 10)}).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps = a.epochs * len(dtr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
def fwd(x):
    with torch.autocast('cuda', torch.bfloat16): h = model['enc'](x.contiguous(memory_format=torch.channels_last))
    h = h.float(); return model['reg'](h), model['c3'](h), model['c10'](h), h
def pearson(p, y):
    p, y = p - p.mean(), y - y.mean(); return float((p * y).sum() / (np.sqrt((p * p).sum() * (y * y).sum()) + 1e-9))
@torch.no_grad()
def evaluate(save=False):
    model.eval(); P, C3, G = [], [], []
    for raw, ph, g, y, m, k3, k10 in dev:
        p, l3, _, h = fwd(prep(raw, ph, g, False)); P.append(p.cpu().numpy()); C3.append(l3.argmax(1).cpu().numpy()); G.append(g.numpy())
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
    if save: np.savez(a.out + '/eval_preds.npz', P=P, G=G, C3=C3)
    model.train(); return r
try:
    import wandb; run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), name=os.path.basename(a.out.rstrip('/')), id=os.path.basename(a.out.rstrip('/')), resume='allow', config=vars(a))
except Exception as e: print('wandb off:', e); run = None
ck = a.out + '/ckpt.pt'; ep0 = 0; log = []
if os.path.exists(ck):
    s = torch.load(ck, map_location='cuda', weights_only=False)
    cfg = {k: v for k, v in vars(a).items() if k not in ('workers', 'out')}
    if s.get('cfg') != cfg or not np.array_equal(s.get('tr'), tr) or not np.array_equal(s.get('ev'), ev):
        raise SystemExit(f'refusing to resume {ck}: config/rows differ from the checkpoint (red-team F3)')
    model.load_state_dict(s['m']); opt.load_state_dict(s['o']); sched.load_state_dict(s['s']); ep0, log = s['ep'], s['log']; print('resumed', ep0, flush=True)
t0 = time.time(); step = ep0 * len(dtr)
for ep in range(ep0, a.epochs):
    tl = n = 0
    for raw, ph, g, y, m, k3, k10 in dtr:
        y, m, k3, k10 = y.cuda(), m.cuda(), k3.cuda(), k10.cuda(); p, l3, l10, _ = fwd(prep(raw, ph, g, True))
        reg = (((p - y) ** 2) * m).sum(0) / m.sum(0).clamp(min=1); reg = reg[m.any(0)].mean()
        ce = (F.cross_entropy(l3, k3, ignore_index=-1) + F.cross_entropy(l10, k10, ignore_index=-1)) if (k3 >= 0).any() else p.sum() * 0
        loss = reg + a.w_cls * ce; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        tl += loss.item(); n += 1; step += 1
        if step % 200 == 0:
            print(f'ep {ep} step {step}/{steps} loss {tl/n:.4f} {time.time()-t0:.0f}s', flush=True)
            if run: run.log({'train/loss': tl / n}, step=step)
    r = evaluate(save=(ep + 1 == a.epochs)); r.update(epoch=ep + 1, train_loss=tl / n, secs=time.time() - t0); log.append(r); print(json.dumps(r), flush=True)
    if run: run.log({f'eval/{k}': v for k, v in r.items()}, step=step)
    torch.save({'m': model.state_dict(), 'o': opt.state_dict(), 's': sched.state_dict(), 'ep': ep + 1, 'log': log, 'tr': tr, 'ev': ev,
                'cfg': {k: v for k, v in vars(a).items() if k not in ('workers', 'out')}}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
if run: run.finish()
json.dump({'args': vars(a), 'final': log[-1], 'log': log, 'n_train': len(tr), 'n_eval': len(ev), 'targets': names}, open(a.out + '/result.json', 'w'), indent=1)
