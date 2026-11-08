"""LF->markers+celltype baseline. usage: train_lf.py --channels all --arch resnet18 --epochs 8 --out DIR [--max-train N]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, json, time, numpy as np, torch, torch.nn as nn, timm
from bsccm.data import LEDMarkerSet, chan_list, W
ap = argparse.ArgumentParser(); ap.add_argument('--channels', default='all'); ap.add_argument('--arch', default='resnet18')
ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--workers', type=int, default=16); ap.add_argument('--out', required=True); ap.add_argument('--max-train', type=int, default=0)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--eval-split', type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
T = np.load(W + '/targets.npz'); names = list(T['names']); K = len(names); NC = int(T['cls'].max()) + 1
has = T['mask'].any(1) | (T['cls'] >= 0); sp = T['split']
tr = np.where((sp == 0) & has)[0]; ev = np.where((sp == a.eval_split) & has)[0]
if a.max_train: tr = np.random.default_rng(0).permutation(tr)[:a.max_train]
# target standardization from train valid entries
Y, Mk = T['Y'], T['mask']; ym = np.array([Y[tr, j][Mk[tr, j]].mean() for j in range(K)], np.float32); ys = np.array([Y[tr, j][Mk[tr, j]].std() + 1e-6 for j in range(K)], np.float32)
dl = lambda rows, train: torch.utils.data.DataLoader(LEDMarkerSet(rows, a.channels, train=train), batch_size=a.bs, shuffle=train, num_workers=a.workers, drop_last=train, pin_memory=True, persistent_workers=True, prefetch_factor=4)
dtr, dev = dl(tr, True), dl(ev, False)
net = timm.create_model(a.arch, pretrained=False, num_classes=0, in_chans=len(chan_list(a.channels)))
head = nn.Linear(net.num_features, K + NC); model = nn.Sequential(net, head).cuda().to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); steps = a.epochs * len(dtr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
ym_t, ys_t = torch.tensor(ym).cuda(), torch.tensor(ys).cuda()
def evaluate():
    model.eval(); P, Yt, Mt, C, Cp = [], [], [], [], []
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        for x, y, m, c, *_ in dev:
            o = model(x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)).float()
            P.append((o[:, :K] * ys_t + ym_t).cpu()); Yt.append(y); Mt.append(m); C.append(c); Cp.append(o[:, K:].argmax(1).cpu())
    P, Yt, Mt, Cp = torch.cat(P), torch.cat(Yt), torch.cat(Mt), torch.cat(Cp); C = torch.as_tensor(np.concatenate([np.asarray(c) for c in C]))
    r = {}
    for j, n in enumerate(names):
        v = Mt[:, j]
        if v.sum() > 10: r[n] = float(torch.corrcoef(torch.stack([P[v, j], Yt[v, j]]))[0, 1])
    lab = C >= 0; r['cls_acc'] = float((Cp[lab] == C[lab]).float().mean()) if lab.any() else None
    if lab.any(): r['cls_bal_acc'] = float(np.mean([(Cp[lab & (C == k)] == k).float().mean().item() for k in range(NC) if (lab & (C == k)).any()]))
    r['pearson_mean'] = float(np.mean([v for k, v in r.items() if ':' in k])); model.train(); return r
try:
    import wandb; run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), name=os.path.basename(a.out.rstrip('/')), id=os.path.basename(a.out.rstrip('/')), resume='allow', config=vars(a))
except Exception as e: print('wandb off:', e); run = None
t0 = time.time(); log = []; step = 0
ck = a.out + '/ckpt.pt'; ep0 = 0
if os.path.exists(ck): s = torch.load(ck, map_location='cuda'); model.load_state_dict(s['m']); opt.load_state_dict(s['o']); sched.load_state_dict(s['s']); ep0 = s['ep']; log = s['log']; print('resumed at epoch', ep0, flush=True)
for ep in range(ep0, a.epochs):
    model.train(); tl = 0; n = 0
    for x, y, m, c, *_ in dtr:
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.cuda(); m = m.cuda(); c = c.cuda()
        with torch.autocast('cuda', torch.bfloat16): o = model(x).float()
        reg = (((o[:, :K] - (y - ym_t) / ys_t) ** 2) * m).sum(0) / m.sum(0).clamp(min=1); reg = reg[m.any(0)].mean() if m.any() else o.sum() * 0
        ce = nn.functional.cross_entropy(o[:, K:], c, ignore_index=-1) if (c >= 0).any() else o.sum() * 0
        loss = reg + 0.5 * ce; opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step()
        tl += loss.item(); n += 1; step += 1
        if step % 100 == 0:
            print(f'ep {ep} step {step}/{steps} loss {tl/n:.4f} {time.time()-t0:.0f}s', flush=True)
            if run: run.log({'train/loss': tl / n, 'lr': sched.get_last_lr()[0]}, step=step)
    r = evaluate(); r.update(epoch=ep + 1, train_loss=tl / n, secs=time.time() - t0); log.append(r); print(json.dumps(r), flush=True)
    if run: run.log({f'eval/{k}': v for k, v in r.items() if isinstance(v, (int, float))}, step=step)
    torch.save({'m': model.state_dict(), 'o': opt.state_dict(), 's': sched.state_dict(), 'ep': ep + 1, 'log': log}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
if run: run.finish()
json.dump({'args': vars(a), 'final': log[-1], 'log': log, 'n_train': len(tr), 'n_eval': len(ev), 'targets': names}, open(a.out + '/result.json', 'w'), indent=1)
