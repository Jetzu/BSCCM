"""N033 data engine: render virtual Wright images with a trained vs_train.py generator for arbitrary LF rows (e.g. all marker-labelled,
non-Giemsa, training-FOV cells). Writes <out>/images.npy (N,128,128,3) uint8 (same units as the generator target: 1.0 = rgb_scale) and
<out>/rows.npy (global LF rows), <out>/meta.json. Resumable (memmap + cursor file), SIGUSR1/SIGTERM-safe.
Usage: vs_generate.py --checkpoint ckpt.pt --registration register.npz --targets targets_v6c.npz --out DIR [--split 0]"""
import sys, os, json, signal, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from vs_common import W, UNet, amp, make_loader
from bsccm.gpu_norm import RawSet, GPUNorm
ap = argparse.ArgumentParser(); ap.add_argument('--checkpoint', required=True); ap.add_argument('--registration', required=True)
ap.add_argument('--targets', default='targets_v6c.npz'); ap.add_argument('--split', type=int, default=0); ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=256); ap.add_argument('--workers', type=int, default=15); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
stop = False
def _sig(*_):
    global stop; stop = True
signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGUSR1, _sig)
T = np.load(W + '/' + a.targets, allow_pickle=True)
rows = np.where((T['hist_row'] < 0) & T['mask'].any(1) & (T['split'] == a.split))[0]   # marker/autofl-labelled, never a Giemsa cell
saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False); cfg = saved['config']
net = UNet(cfg['width'], cfg['out_size']).cuda().to(memory_format=torch.channels_last); net.load_state_dict(saved['model'], strict=True); net.eval()
rgb_scale = float(np.load(a.registration)['rgb_scale']); size = cfg['out_size']
ip, cp = a.out + '/images.npy', a.out + '/cursor.json'
if os.path.exists(cp):
    st = json.load(open(cp)); assert st['n'] == len(rows) and st['checkpoint'] == a.checkpoint, 'resume mismatch'; cursor = st['cursor']
    imgs = np.lib.format.open_memmap(ip, mode='r+')
else:
    cursor = 0; imgs = np.lib.format.open_memmap(ip, mode='w+', dtype=np.uint8, shape=(len(rows), size, size, 3)); np.save(a.out + '/rows.npy', rows)
norm = GPUNorm('cuda'); t0 = time.time(); last = t0
dl = torch.utils.data.DataLoader(RawSet(rows[cursor:]), batch_size=a.bs, shuffle=False, num_workers=a.workers, pin_memory=True, prefetch_factor=4)
with torch.no_grad():
    for raw, ph, g in dl:
        x = norm(raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True), ch='all', phase=True, shift=0).contiguous(memory_format=torch.channels_last)
        with amp('cuda'): y = net(x)
        q = (y.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        imgs[cursor:cursor + len(q)] = q; cursor += len(q)
        if stop or time.time() - last > 600 or cursor == len(rows):
            imgs.flush(); json.dump(dict(n=int(len(rows)), cursor=int(cursor), checkpoint=a.checkpoint), open(cp + '.tmp', 'w')); os.replace(cp + '.tmp', cp); last = time.time()
            print(f'{cursor}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
        if stop: sys.exit(0)
json.dump(dict(n=int(len(rows)), rgb_scale=rgb_scale, size=size, split=a.split, targets=a.targets, checkpoint=a.checkpoint, done=True), open(a.out + '/meta.json', 'w'))
print('done', len(rows), f'{time.time() - t0:.0f}s')
