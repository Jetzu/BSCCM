"""Held-out virtual-stain evaluation, resumable generation and student-ready NPZ."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import io
import json
import os
import signal
import time
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from vs_common import (W, VERSION, UNet, Dino, Pairs, GPUNorm, amp, atomic_npz,
    digest, image_metrics, make_loader, normalize_batch, read_registration, retrieval)


def export_npz(path, image_file, **arrays):
    """Stream the large .npy into an uncompressed NPZ; no full image RAM copy."""
    with zipfile.ZipFile(str(path)+'.tmp', 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.write(image_file, 'images.npy')
        for key, value in arrays.items():
            buf = io.BytesIO(); np.save(buf, value, allow_pickle=False)
            z.writestr(key+'.npy', buf.getvalue())
    with open(str(path)+'.tmp', 'rb') as f:
        os.fsync(f.fileno())
    os.replace(str(path)+'.tmp', path)


def montage(path, images, dataset, reg_rows, seed):
    from PIL import Image, ImageDraw
    chosen = np.random.default_rng(seed).choice(len(dataset), min(24, len(dataset)), replace=False)
    size = images.shape[1]; label_h = 22
    canvas = Image.new('RGB', (size*3, (size+label_h)*len(chosen)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, j in enumerate(chosen):
        raw, _, g, real = dataset[int(j)]
        real = (real.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        bf = raw[0].numpy().astype(np.float32)
        lo, hi = np.percentile(bf, [1, 99])
        bf = ((bf-lo)/max(hi-lo, 1e-6)*255).clip(0, 255).astype(np.uint8)
        bf = Image.fromarray(bf).convert('RGB').resize((size, size))
        y = i*(size+label_h)
        draw.text((4, y+3), f'global {g} hist {int(reg_rows[j])} | real / generated / LF BF', fill='black')
        for k, image in enumerate([Image.fromarray(real), Image.fromarray(np.asarray(images[j])), bf]):
            canvas.paste(image, (k*size, y+label_h))
    canvas.save(path)
    return chosen


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--registration', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pack', default=W+'/pack')
    p.add_argument('--targets', default=W+'/targets_v6.npz')
    p.add_argument('--dino-b', default=W+'/models/DinoBloom-B.pth')
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if a.bs < 1 or a.workers < 0:
        p.error('invalid batch/workers')
    stop = False
    def on_signal(signum, frame):
        nonlocal stop
        stop = True
    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, on_signal)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']
    if cfg['version'] != VERSION or cfg['registration_sha256'] != digest(a.registration):
        raise ValueError('checkpoint registration/version mismatch')
    if cfg['targets_sha256'] != digest(a.targets):
        raise ValueError('checkpoint targets mismatch')
    if cfg['normalizer_sha256'] != {name: digest(Path(W)/name) for name in cfg['normalizer_sha256']}:
        raise ValueError('normalizer metadata changed since training')
    reg, _, te = read_registration(a.registration, a.targets, a.pack, cfg['holdout'])
    size, n = cfg['out_size'], len(te)
    run_cfg = dict(version=VERSION, checkpoint_sha256=digest(a.checkpoint),
                   registration_sha256=digest(a.registration), dino_b_sha256=digest(a.dino_b),
                   targets_sha256=digest(a.targets), seed=a.seed, n=n, size=size)
    final_meta_path = out/'metrics.json'
    final_images = out/'generated.npz'
    if final_meta_path.exists() and final_images.exists():
        old = json.loads(final_meta_path.read_text())
        if old['config'] != run_cfg:
            raise ValueError('evaluation output belongs to different config; use a fresh --out')
        print(json.dumps(dict(event='already_complete', metrics=str(final_meta_path))), flush=True)
        return
    net = UNet(cfg['width'], size).to(a.device, memory_format=torch.channels_last)
    net.load_state_dict(saved['model'], strict=True); net.eval()
    dino = Dino('B', a.dino_b).to(a.device)
    norm = GPUNorm(a.device)
    state_path, image_path = out/'eval_state.npz', out/'generated.partial.npy'
    if state_path.exists():
        with np.load(state_path) as f:
            if json.loads(str(f['config'])) != run_cfg:
                raise ValueError('evaluation resume config mismatch; use a fresh --out')
            cursor = int(f['cursor'])
            values = {k: f[k] for k in ['psnr', 'ssim', 'cosine', 'real_embed', 'gen_embed',
                                       'real_mean', 'real_second', 'gen_mean', 'gen_second']}
        images = np.lib.format.open_memmap(image_path, mode='r+')
        if images.shape != (n, size, size, 3) or images.dtype != np.uint8:
            raise ValueError('invalid generated image resume file')
    else:
        cursor = 0
        images = np.lib.format.open_memmap(image_path, mode='w+', dtype=np.uint8, shape=(n, size, size, 3))
        values = dict(psnr=np.zeros(n), ssim=np.zeros(n), cosine=np.zeros(n),
                      real_embed=np.zeros((n, 768), np.float32), gen_embed=np.zeros((n, 768), np.float32),
                      **{k: np.zeros((n, 3), np.float64) for k in
                         ['real_mean', 'real_second', 'gen_mean', 'gen_second']})
    def save():
        images.flush()
        with open(image_path, 'rb') as f:
            os.fsync(f.fileno())
        atomic_npz(state_path, cursor=cursor, config=json.dumps(run_cfg), **values)
    save()
    loader = make_loader(Pairs(reg, te[cursor:], size, a.pack), a.bs, a.workers)
    last_save = time.monotonic()
    for batch in loader:
        if stop:
            break
        x, real = normalize_batch(batch, norm, a.device)
        with amp(a.device):
            pred = net(x)
        # Metrics, embeddings and downstream student all see the same quantized RGB.
        q = (pred.float()*255).round().clamp(0, 255).to(torch.uint8)
        pred = q.float()/255
        with amp(a.device):
            re = F.normalize(dino(real).float(), dim=-1)
            ge = F.normalize(dino(pred).float(), dim=-1)
        psnr, ssim = image_metrics(pred, real)
        sl = slice(cursor, cursor+len(x))
        images[sl] = q.permute(0, 2, 3, 1).cpu().numpy()
        values['psnr'][sl], values['ssim'][sl] = psnr.cpu().numpy(), ssim.cpu().numpy()
        values['cosine'][sl] = (re*ge).sum(-1).cpu().numpy()
        values['real_embed'][sl], values['gen_embed'][sl] = re.cpu().numpy(), ge.cpu().numpy()
        for key, image in [('real', real), ('gen', pred)]:
            values[key+'_mean'][sl] = image.mean((2, 3)).cpu().numpy()
            values[key+'_second'][sl] = image.square().mean((2, 3)).cpu().numpy()
        cursor += len(x)
        if stop or time.monotonic()-last_save > 300:
            save(); last_save = time.monotonic()
    save()
    if stop:
        print(json.dumps(dict(event='stopped', completed=cursor, total=n)), flush=True)
        return
    if cursor != n:
        raise RuntimeError('incomplete evaluation cursor')
    rng = np.random.default_rng(a.seed)
    candidates = np.sort(rng.choice(n, min(1000, n), replace=False))
    color = {}
    for key in ['real', 'gen']:
        mean = values[key+'_mean'].mean(0)
        std = np.sqrt(np.maximum(0, values[key+'_second'].mean(0)-mean**2))
        color[key] = dict(mean=mean.tolist(), std=std.tolist(), units='RGB in [0,1]')
    # Report each batch separately, because pooled slide colour can flatter scores.
    with np.load(a.targets) as t:
        batches = t['batch'][reg['gidx'][te]]
    summaries = {}
    for b in np.unique(batches):
        mask = batches == b
        summaries[str(int(b))] = dict(n=int(mask.sum()),
            **{k: float(values[k][mask].mean()) for k in ['psnr', 'ssim', 'cosine']})
    ds = Pairs(reg, te, size, a.pack)
    montage_rows = montage(out/'montage.png', images, ds, reg['hist_row'][te], a.seed)
    meta = dict(config=run_cfg, n_heldout=n, holdout=cfg['holdout'], step=saved['step'],
                epoch=saved['epoch'], metrics_on='uint8 generated RGB vs registered float real RGB',
                psnr=float(values['psnr'].mean()), ssim=float(values['ssim'].mean()),
                dino_b_cosine=float(values['cosine'].mean()), per_batch=summaries,
                generated_to_real_retrieval=retrieval(values['gen_embed'][candidates], values['real_embed'][candidates]),
                real_to_generated_retrieval=retrieval(values['real_embed'][candidates], values['gen_embed'][candidates]),
                color=color, registration_report=json.loads(str(reg['meta']))['report'])
    atomic_npz(out/'per_cell.npz', gidx=reg['gidx'][te], hist_row=reg['hist_row'][te],
               psnr=values['psnr'], ssim=values['ssim'], cosine=values['cosine'], batch=batches,
               retrieval_gidx=reg['gidx'][te[candidates]], montage_gidx=reg['gidx'][te[montage_rows]])
    export_npz(final_images, image_path, gidx=reg['gidx'][te], hist_row=reg['hist_row'][te],
               rgb_scale=reg['rgb_scale'], out_size=np.array(size), meta=np.array(json.dumps(meta)))
    with open(str(final_meta_path)+'.tmp', 'w') as f:
        json.dump(meta, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(str(final_meta_path)+'.tmp', final_meta_path)
    del images
    image_path.unlink(); state_path.unlink()
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
