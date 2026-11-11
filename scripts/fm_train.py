"""N034 trainer: LF (+ stain token) -> 6-channel fluorescence image. --mode cfm: conditional (rectified) flow matching, x_t =
(1-t) z + t y, z ~ N(0, I), t ~ logit-normal, loss ||v - (y - z)||^2, LF dropped with p --cfg-drop (classifier-free guidance).
--mode reg: the SAME network as a deterministic regressor (x_t = 0, t = 0) with MSE in target space (posterior-mean baseline).
Train rows: batch --train-batch, split 0, never a Giemsa cell (hist_row < 0). Val: same batch split 1 (fixed 2048 cells, fixed noise).
Preempt-safe: resumes <out>/ckpt.pt (model, EMA, optimiser, cursor, RNG), saves every --save-minutes and on SIGTERM/SIGUSR1.
Usage: fm_train.py --out DIR --mode cfm|reg [--train-batch 1] [--epochs 30]"""
import sys, os, json, time, signal, argparse, hashlib, copy, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from fm_common import (W, FluorSet, Prep, GPUNorm, CondUNet, EpochBatches, make_loader, make_cond, batch_to_device, amp,
                       atomic_torch, rng_state, restore_rng, digest)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', required=True); p.add_argument('--mode', choices=['cfm', 'reg'], default='cfm')
p.add_argument('--targets', default=W + '/targets_v5c.npz'); p.add_argument('--train-batch', type=int, default=1)   # red-team F3: clean lineage
p.add_argument('--epochs', type=int, default=30); p.add_argument('--bs', type=int, default=128); p.add_argument('--lr', type=float, default=2e-4)
p.add_argument('--width', type=int, default=64); p.add_argument('--patch', type=int, default=2); p.add_argument('--cfg-drop', type=float, default=0.1); p.add_argument('--translate', type=int, default=4)
p.add_argument('--ema', type=float, default=0.9995); p.add_argument('--seed', type=int, default=0); p.add_argument('--workers', type=int, default=15)
p.add_argument('--save-minutes', type=float, default=20); p.add_argument('--max-train', type=int, default=0, help='smoke tests only')
a = p.parse_args()
stop = False
def _sig(*_):
    global stop; stop = True
for s in (signal.SIGTERM, signal.SIGUSR1): signal.signal(s, _sig)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); dev = 'cuda'
assert torch.cuda.is_available(), 'no silent CPU training'
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True
T = np.load(a.targets, allow_pickle=True)
base = (T['hist_row'] < 0) & (T['batch'] == a.train_batch)
tr = np.flatnonzero(base & (T['split'] == 0)); va = np.flatnonzero(base & (T['split'] == 1))
va = np.sort(np.random.default_rng(1).choice(va, min(2048, len(va)), replace=False))
if a.max_train: tr = np.sort(np.random.default_rng(2).choice(tr, a.max_train, replace=False))
cfg = {k: v for k, v in vars(a).items() if k not in ('out', 'workers', 'save_minutes')}
cfg.update(prep_sha256=digest(W + '/fm_prep.npz'), targets_sha256=digest(a.targets), n_train=int(len(tr)),
           lf_norm_sha256={n: digest(W + '/' + n) for n in ('cellmeta.npz', 'led_norm.npz', 'backgrounds_p50.npy')})   # red-team F8
ck = out / 'ckpt.pt'; saved = torch.load(ck, map_location='cpu', weights_only=False) if ck.exists() else None
if saved is not None and saved['config'] != cfg:
    raise ValueError(f"resume config mismatch: {[k for k in cfg.keys() | saved['config'].keys() if cfg.get(k) != saved['config'].get(k)]}")
prep = Prep(dev); norm = GPUNorm(dev)
w = a.width; net = CondUNet(ch=(w, 2 * w, 4 * w, 4 * w), n_stain=len(prep.stain_names), patch=a.patch).to(dev, memory_format=torch.channels_last)
ema = copy.deepcopy(net).requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
sampler = EpochBatches(len(tr), a.bs, a.seed); total = a.epochs * sampler.total
lr_at = lambda s: a.lr * min(1., (s + 1) / 1000) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1., s / total))))
epoch = nb = step = 0
if saved is not None:
    net.load_state_dict(saved['model']); ema.load_state_dict(saved['ema']); opt.load_state_dict(saved['opt'])
    epoch, nb, step = saved['epoch'], saved['next_batch'], saved['step']
print(json.dumps(dict(event='start', n_train=len(tr), n_val=len(va), params_M=sum(q.numel() for q in net.parameters()) / 1e6,
                      stains=prep.stain_names, resumed=saved is not None, step=step)), flush=True)
tl = make_loader(FluorSet(tr), a.bs, a.workers, sampler); vl = make_loader(FluorSet(va), 256, 4)
run_id = saved['wandb_id'] if saved is not None else hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:12]
import wandb
wb = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), id=run_id, name=f'N034-{a.mode}-b{a.train_batch}-s{a.seed}',
                mode='offline', config=cfg)
with open(out / 'sync_lineage.jsonl', 'a') as f:
    f.write(json.dumps(dict(wandb_id=run_id, job_id=os.environ.get('SLURM_JOB_ID'), resumed=saved is not None, time=time.time())) + '\n')
if saved is not None: restore_rng(saved['rng'])
last = time.monotonic()
def save():
    global last
    atomic_torch(dict(config=cfg, model=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), epoch=epoch, next_batch=nb,
                      step=step, rng=rng_state(), wandb_id=run_id, stain_names=prep.stain_names), ck)
    last = time.monotonic()

def loss_fn(model, x, y, st, gen=None):
    if a.mode == 'reg':
        with amp(dev):
            pred = model(torch.zeros_like(y), torch.zeros(len(y), device=dev), st, make_cond(x)).float()
        return F.mse_loss(pred, y)
    z = torch.randn(y.shape, device=dev, generator=gen)
    t = torch.sigmoid(torch.randn(len(y), device=dev, generator=gen))
    xt = (1 - t)[:, None, None, None] * z + t[:, None, None, None] * y
    with amp(dev):
        v = model(xt, t, st, make_cond(x, a.cfg_drop if model.training else 0.)).float()
    return F.mse_loss(v, y - z)

@torch.no_grad()
def validate():
    ema.eval(); g = torch.Generator(device=dev).manual_seed(123); s = n = 0
    for batch in vl:
        x, y, st, _ = batch_to_device(batch, norm, prep, dev)
        l = loss_fn(ema, x, y, st, g); s += float(l) * len(x); n += len(x)
    return s / n

save()
while epoch < a.epochs and not stop:
    sampler.epoch, sampler.start = epoch, nb; net.train(); acc = cnt = 0; t0 = time.monotonic()
    for batch in tl:
        x, y, st, _ = batch_to_device(batch, norm, prep, dev, a.translate)
        for gq in opt.param_groups: gq['lr'] = lr_at(step)
        loss = loss_fn(net, x, y, st)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 1 - a.ema)
        step += 1; nb += 1; acc += float(loss.detach()) * len(x); cnt += len(x)
        if step % 100 == 0:
            wb.log({'train/loss': acc / max(cnt, 1), 'lr': lr_at(step), 'grad_norm': float(gn),
                    'cells_per_s': cnt / (time.monotonic() - t0)}, step=step)
            print(json.dumps(dict(step=step, epoch=epoch, loss=acc / max(cnt, 1), cells_s=round(cnt / (time.monotonic() - t0)))), flush=True)
        if stop or time.monotonic() - last > a.save_minutes * 60: save()
        if stop: break
    if stop: break
    assert nb == sampler.total, 'epoch cursor mismatch'
    epoch += 1; nb = 0; vloss = validate()
    log = dict(event='epoch', epoch=epoch, step=step, train_loss=acc / max(cnt, 1), val_loss_ema=vloss, seconds=time.monotonic() - t0)
    print(json.dumps(log), flush=True); wb.log({'val/loss_ema': vloss, 'epoch': epoch}, step=step)
    with open(out / 'metrics.jsonl', 'a') as f: f.write(json.dumps(log) + '\n')
    save()
save(); wb.finish(); print(json.dumps(dict(event='stopped' if stop else 'complete', epoch=epoch, step=step)), flush=True)
