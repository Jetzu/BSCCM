"""N034 evaluation of LF -> fluorescence-image generators (fm_train.py cfm and/or reg checkpoints, EMA weights).
Scored cells (never Giemsa): 'inbatch' = training blood draw, held-out FOVs (split 2); 'newdraw' = the other draw, split 2 ONLY
(red-team F1: that draw's split-0 cells calibrate its own position gains / curated labels, so they are never scored). --per-stain
cells per stain (seeded, own RNG per set x stain).
Arms per cell: K flow samples (Heun --steps, --cfg) and their target-space mean; the regression output; CONTROLS (red-team F4):
'template' = mean target-space image of the stain on readout-fit cells (no LF), 'shuffled' = one flow sample conditioned on the LF
of ANOTHER cell of the same stain/set (unconditional-like generator).
READOUT (image -> curated label): ridge per stain on 24 radial integrals (6 ch x disk/3 annuli of gain-corrected linear counts, asinh),
fitted on REAL images of training-draw split-0 cells -> clean curated targets_v5c label of that stain (single:X; unstained -> autofl;
all -> grp:*). r_realref = r(readout(real image), label) is the real-image reference agreement (not a strict bound).
Per stain x set x label: r_avgread (mean over K of the readouts = scalar conditional-mean estimate; primary), r_sample (one sample),
r_mean (readout of the target-space mean image), r_reg, r_shuffled; distribution vs the REAL readouts: W1/sd (marginal), conditional
W1/sd within quintiles of the regression readout (LF-predictable strata), CRPS/sd vs the real readout (proper score: rewards
calibration AND sharpness; regression = point forecast), sharpness = mean per-cell sample sd / marginal sd, rank histogram.
Spatial: per-channel pixel Pearson inside the r<=56 disk vs the real image for sample, sample mean, regression, template, shuffled.
Checks (red-team F8): checkpoint mode and fm_prep hash. Writes eval.json, per_cell.npz, montage.npz."""
import sys, os, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from fm_common import W, FluorSet, Prep, GPUNorm, CondUNet, make_loader, batch_to_device, sample, regress, digest, Generator

ap = argparse.ArgumentParser(); ap.add_argument('--cfm', default=''); ap.add_argument('--reg', default=''); ap.add_argument('--out', required=True)
ap.add_argument('--targets', default=W + '/targets_v5c.npz'); ap.add_argument('--per-stain', type=int, default=400); ap.add_argument('--K', type=int, default=8)
ap.add_argument('--steps', type=int, default=16); ap.add_argument('--cfg', type=float, default=1.0); ap.add_argument('--bs', type=int, default=200)
ap.add_argument('--readout-per-stain', type=int, default=4000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--newdraw-split', type=int, default=2, help='split of the other draw that is scored (never 0: its calibration cells)')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = 'cuda'
T = np.load(a.targets, allow_pickle=True); names = [str(n) for n in T['names']]; Y, M = T['Y'], T['mask']
prep = Prep(dev); norm = GPUNorm(dev); SN = prep.stain_names; stain = np.load(W + '/fm_prep.npz')['stain'].astype(int)
prep_sha = digest(W + '/fm_prep.npz')
def load(path, mode):
    s = torch.load(path, map_location='cpu', weights_only=False); c = s['config']; w = c['width']
    assert c['mode'] == mode or (mode == 'cfm' and c['mode'] == 'rcfm'), f'{path}: checkpoint mode {c["mode"]} != {mode}'
    assert c['prep_sha256'] == prep_sha, f'{path}: trained on a different fm_prep.npz'
    for n_, h_ in c.get('lf_norm_sha256', {}).items(): assert digest(W + '/' + n_) == h_, f'{path}: LF normalisation file {n_} changed'   # rereview F8
    net = CondUNet(ch=(w, 2 * w, 4 * w, 4 * w), n_stain=len(SN), patch=c.get('patch', 1), ccond=25 + (6 if c['mode'] == 'rcfm' else 0)).to(dev)
    net.load_state_dict(s['ema']); net.eval(); assert s['stain_names'] == SN
    if c['mode'] == 'rcfm':   # N044: wrap with the frozen mean regressor recorded in the checkpoint (hash-checked)
        assert digest(c['mean_ckpt']) == c['mean_ckpt_sha256'], 'rcfm mean checkpoint changed since training'
        mnet, _ = load(c['mean_ckpt'], 'reg'); return Generator(net, mnet, s['res_scale']), c
    return (Generator(net) if mode == 'cfm' else net), c
nets, cfgs, tb = {}, {}, None
for k in ('cfm', 'reg'):
    if getattr(a, k):
        nets[k], cfgs[k] = load(getattr(a, k), k); assert tb in (None, cfgs[k]['train_batch']); tb = cfgs[k]['train_batch']
LAB = {s: [f'single:{s}'] for s in SN if s not in ('all', 'unstained')}; LAB['unstained'] = ['autofl']
LAB['all'] = ['grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16', 'grp:CD123/HLA-DR/CD14']
yy, xx = np.mgrid[:128, :128]; R = np.hypot(yy - 63.5, xx - 63.5)
MASKS = torch.tensor(np.stack([R <= 56, R < 16, (R >= 16) & (R < 32), (R >= 32) & (R <= 56)]), dtype=torch.float32, device=dev)
DISK = MASKS[0].bool()
def feats(y):   # target-space images (B,6,128,128) -> (B,24) asinh of radial integrals of linear counts
    I = torch.einsum('bcyx,kyx->bck', prep.linear(y.float()), MASKS)
    return torch.asinh(I / (prep.a[0, :, 0, 0, None] * 100)).flatten(1).cpu().numpy()
def pix_r(p, y):   # per-cell mean over channels of the Pearson r inside the disk
    p, y = p.float()[:, :, DISK], y.float()[:, :, DISK]; p = p - p.mean(-1, keepdim=True); y = y - y.mean(-1, keepdim=True)
    r = (p * y).sum(-1) / (p.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-8); return r.mean(1).cpu().numpy()
pear = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float('nan')
w1 = lambda u, v: float(np.mean(np.abs(np.sort(u) - np.sort(v))))
def crps(S, y):   # S (n,K) ensemble, y (n,) -> mean CRPS
    return float(np.mean(np.abs(S - y[:, None]).mean(1) - 0.5 * np.abs(S[:, :, None] - S[:, None, :]).mean((1, 2))))

clean = T['hist_row'] < 0
readout, templ = {}, {}
for si, s in enumerate(SN):   # readout + template on REAL training-draw split-0 images only
    rng = np.random.default_rng([a.seed, 0, si])
    rows = np.flatnonzero(clean & (T['batch'] == tb) & (T['split'] == 0) & (stain == si))
    rows = np.sort(rng.choice(rows, min(a.readout_per_stain, len(rows)), replace=False)); F_, acc = [], 0
    with torch.no_grad():
        for batch in make_loader(FluorSet(rows), 256, 12):
            _, y, _, _ = batch_to_device(batch, norm, prep, dev); F_.append(feats(y)); acc = acc + y.sum(0)
    Fr = np.concatenate(F_); templ[s] = acc / len(rows); mu, sd = Fr.mean(0), Fr.std(0) + 1e-6; Z = (Fr - mu) / sd
    for lab in LAB[s]:
        j = names.index(lab); ok = M[rows, j]
        if ok.sum() < 200: continue
        A = np.c_[Z[ok], np.ones(ok.sum())]; wgt = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ Y[rows[ok], j])
        readout[(s, lab)] = (mu, sd, wgt)
apply_ro = lambda F_, key: np.c_[(F_ - readout[key][0]) / readout[key][1], np.ones(len(F_))] @ readout[key][2]

assert a.newdraw_split in (1, 2)
res = dict(train_batch=int(tb), newdraw_split=a.newdraw_split, K=a.K, steps=a.steps, cfg=a.cfg, seed=a.seed, per_stain=a.per_stain, readout_per_stain=a.readout_per_stain, bs=a.bs,
           readout={f'{k[0]}|{k[1]}': dict(mu=v[0].tolist(), sd=v[1].tolist(), w=v[2].tolist()) for k, v in readout.items()}, targets=a.targets, targets_sha256=digest(a.targets), prep_sha256=prep_sha,
           ckpt={k: getattr(a, k) for k in nets}, train_cfg=cfgs, sets={}); per_cell = {}; montage = {}
for set_name, base in (('inbatch', clean & (T['batch'] == tb) & (T['split'] == 2)), ('newdraw', clean & (T['batch'] != tb) & (T['split'] == a.newdraw_split))):
    res['sets'][set_name] = {}
    for si, s in enumerate(SN):
        part = f'{a.out}/part_{set_name}_{s}.npy'   # preemption-safe: finished groups are reloaded (every group has its own seeds)
        if os.path.exists(part):
            P_ = np.load(part, allow_pickle=True).item(); res['sets'][set_name][s] = P_['r']; per_cell.update(P_['per_cell'])
            if P_['montage'] is not None: montage[f'{set_name}|{s}'] = P_['montage']
            print('resumed', set_name, s, flush=True); continue
        rng = np.random.default_rng([a.seed, 1 if set_name == 'inbatch' else 2, si])
        rows = np.flatnonzero(base & (stain == si)); rows = np.sort(rng.choice(rows, min(a.per_stain, len(rows)), replace=False))
        g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + si + (500 if set_name == 'newdraw' else 0))
        pr = rng.permutation(len(rows))   # LF of ANOTHER cell of the same stain/set: derangement (rereview N6: no fixed points)
        assert len(pr) >= 2, 'shuffled control needs >= 2 cells'
        while (fx := np.flatnonzero(pr == np.arange(len(pr)))).size:   # swap a fixed point with its neighbour: stays a permutation,
            i = fx[0]; j = (i + 1) % len(pr); pr[[i, j]] = pr[[j, i]]   # removes >= 1 fixed point, creates none (check3 F7)
        assert np.array_equal(np.sort(pr), np.arange(len(pr))) and not (pr == np.arange(len(pr))).any()
        perm = torch.tensor(pr, device=dev)
        out = {k: [] for k in ('real', 'reg', 'samp', 'mean', 'shuf', 'tmpl', 'px_samp', 'px_mean', 'px_reg', 'px_tmpl', 'px_shuf')}
        xs_all = []
        with torch.no_grad():
            for bi, batch in enumerate(make_loader(FluorSet(rows), a.bs, 8)):   # unshuffled: order == rows
                x, y, st, _ = batch_to_device(batch, norm, prep, dev); xs_all.append(x.cpu()); out['real'].append(feats(y))
                tp = templ[s][None].expand_as(y); out['px_tmpl'].append(pix_r(tp, y))
                if 'reg' in nets:
                    yr = regress(nets['reg'], x, st); out['reg'].append(feats(yr)); out['px_reg'].append(pix_r(yr, y))
                if 'cfm' in nets:
                    S = torch.stack([nets['cfm'](x, st, a.steps, a.cfg, g) for _ in range(a.K)], 1)
                    out['samp'].append(np.stack([feats(S[:, k]) for k in range(a.K)], 1)); ym = S.mean(1); out['mean'].append(feats(ym))
                    out['px_samp'].append(pix_r(S[:, 0], y)); out['px_mean'].append(pix_r(ym, y))
                    if bi == 0:
                        montage[f'{set_name}|{s}'] = dict(rows=rows[:6], real=y[:6].half().cpu().numpy(), samp=S[:6, :3].half().cpu().numpy(),
                                                          mean=ym[:6].half().cpu().numpy(), reg=yr[:6].half().cpu().numpy() if 'reg' in nets else None,
                                                          lf=x[:6, [0, 5, 23]].half().cpu().numpy(), templ=templ[s].half().cpu().numpy())
            if 'cfm' in nets:   # shuffled-LF control: second pass with permuted LF (same stain, same cells' targets)
                X = torch.cat(xs_all)[perm.cpu()]; Fsh, Psh = [], []
                for bi, batch in enumerate(make_loader(FluorSet(rows), a.bs, 8)):
                    _, y, st, _ = batch_to_device(batch, norm, prep, dev); xb = X[bi * a.bs: bi * a.bs + len(y)].to(dev)
                    ys = nets['cfm'](xb, st, a.steps, a.cfg, g); Fsh.append(feats(ys)); Psh.append(pix_r(ys, y))
                out['shuf'], out['px_shuf'] = Fsh, Psh
        cat = lambda k: np.concatenate(out[k]) if out[k] else None
        Fr, Fg, Fs, Fm, Fsh = cat('real'), cat('reg'), cat('samp'), cat('mean'), cat('shuf'); r = dict(n=int(len(rows)))
        r['pix_r'] = {k[3:]: float(np.nanmean(cat(k))) for k in ('px_samp', 'px_mean', 'px_reg', 'px_tmpl', 'px_shuf') if out[k]}
        for lab in LAB[s]:
            if (s, lab) not in readout: continue
            j = names.index(lab); ok = M[rows, j]
            if ok.sum() < 50: continue
            y_ = Y[rows[ok], j]; ro_real = apply_ro(Fr, (s, lab)); sdr = ro_real.std(); e = dict(n=int(ok.sum()), r_realref=pear(ro_real[ok], y_))
            e['w1_template'] = w1(np.full(len(ro_real), float(apply_ro(feats(templ[s][None]), (s, lab))[0])), ro_real) / sdr
            pc = dict(rows=rows, ok=ok, ro_real=ro_real)
            if Fg is not None:
                ro_reg = apply_ro(Fg, (s, lab)); strata = np.empty(len(ro_reg), int); strata[np.argsort(ro_reg, kind='stable')] = np.arange(len(ro_reg)) * 5 // len(ro_reg)
                pc['ro_reg'] = ro_reg   # rank-based equal-size quintiles: never empty under ties (rereview N6)
                cw1 = lambda v: float(np.mean([w1(v[strata == k], ro_real[strata == k]) for k in range(5)]) / sdr)
                e.update(r_reg=pear(ro_reg[ok], y_), w1_reg=w1(ro_reg, ro_real) / sdr, condw1_reg=cw1(ro_reg), crps_reg=float(np.mean(np.abs(ro_reg - ro_real)) / sdr),
                         r_reg_vs_realref=pear(ro_reg, ro_real))
            if Fs is not None:
                ro_s = np.stack([apply_ro(Fs[:, k], (s, lab)) for k in range(a.K)], 1); ro_m = apply_ro(Fm, (s, lab)); ro_sh = apply_ro(Fsh, (s, lab))
                rank = (ro_s < ro_real[:, None]).sum(1); hist = np.bincount(rank, minlength=a.K + 1) / len(rank); pc.update(ro_s=ro_s, ro_sh=ro_sh)
                e.update(r_avgread=pear(ro_s[ok].mean(1), y_), r_sample=float(np.mean([pear(ro_s[ok, k], y_) for k in range(a.K)])), r_mean=pear(ro_m[ok], y_),
                         r_shuffled=pear(ro_sh[ok], y_), r_avgread_vs_realref=pear(ro_s.mean(1), ro_real),
                         w1_sample=float(np.mean([w1(ro_s[:, k], ro_real) for k in range(a.K)])) / sdr, w1_shuffled=w1(ro_sh, ro_real) / sdr,
                         crps_sample=crps(ro_s, ro_real) / sdr, sharpness=float(ro_s.std(1).mean() / sdr),
                         sd_ratio_sample=float(ro_s.std(0).mean() / sdr), frac_outside=float(hist[0] + hist[-1]), ideal_outside=2 / (a.K + 1), rank_hist=hist.round(3).tolist())
                if Fg is not None:
                    e.update(condw1_sample=float(np.mean([cw1(ro_s[:, k]) for k in range(a.K)])), condw1_shuffled=cw1(ro_sh))
            r[lab] = e; per_cell[f'{set_name}|{s}|{lab}'] = pc
        res['sets'][set_name][s] = r
        with open(part + '.tmp', 'wb') as fh:
            np.save(fh, np.array(dict(r=r, per_cell={k: v for k, v in per_cell.items() if k.startswith(f'{set_name}|{s}|')},
                                      montage=montage.get(f'{set_name}|{s}')), dtype=object), allow_pickle=True)
        os.replace(part + '.tmp', part)
        print(set_name, s, json.dumps({k: ({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'rank_hist'} if isinstance(v, dict) else v)
                                       for k, v in r.items()}), flush=True)
json.dump(res, open(a.out + '/eval.json', 'w'), indent=1)
np.save(a.out + '/per_cell.npy', np.array(per_cell, dtype=object), allow_pickle=True)
np.save(a.out + '/montage.npy', np.array(montage, dtype=object), allow_pickle=True)
print('done')
