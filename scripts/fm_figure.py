"""N034 figure: LF -> fluorescence marker images. (a) new-blood-draw cells (brightest / dimmest of the first six evaluated cells per
stain in the displayed channel = illustration, not marker calls): brightfield, real marker image, 3 flow samples, flow mean,
regression, stain template (same colour scale per row, the stain's dominant channel); (b) per-marker r of the readout vs the curated
label (flow = mean of K readouts, regression, LF-shuffled control; grey tick = real-image reference); (c) distribution scores vs
the real readouts (lower = better). Usage: fm_figure.py --eval DIR --out PNG"""
import sys, json, argparse
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('--eval', required=True); ap.add_argument('--out', required=True); ap.add_argument('--set', default='newdraw')
a = ap.parse_args()
E = json.load(open(a.eval + '/eval.json')); MT = np.load(a.eval + '/montage.npy', allow_pickle=True).item()
BLUE, ORANGE, AQUA, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#9a9a96', '#2b2b29', '#6b6b67'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c8c8c4', 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
SHOW = [('CD16', 'CD16'), ('CD14', 'CD14'), ('CD3', 'CD3'), ('unstained', 'autofl.')]
cols = ['brightfield', 'real', 'sample 1', 'sample 2', 'sample 3', 'flow mean', 'regression', 'template']
fig = plt.figure(figsize=(12.5, 11.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=.22, wspace=.28)
ga = gs[0, :].subgridspec(2 * len(SHOW), len(cols), wspace=.04, hspace=.06)
for i, (st, lab) in enumerate(SHOW):
    m = MT.get(f'{a.set}|{st}')
    if m is None: continue
    real = m['real'].astype(np.float32); ch = int(np.argmax(real[:, :, 32:96, 32:96].mean((2, 3)).std(0)))   # stain's most variable channel
    tot = real[:, ch].mean((1, 2)); picks = [int(np.argmax(tot)), int(np.argmin(tot))]                       # brightest + dimmest shown cell
    for k, c in enumerate(picks):
        imgs = [m['lf'][c, 0].astype(np.float32), real[c, ch]] + [m['samp'][c, q, ch].astype(np.float32) for q in range(3)] + \
               [m['mean'][c, ch].astype(np.float32), m['reg'][c, ch].astype(np.float32) if m['reg'] is not None else np.zeros((128, 128)),
                m['templ'][ch].astype(np.float32)]
        lo, hi = np.percentile(real[:, ch], [1, 99.7])
        for j, im in enumerate(imgs):
            ax = fig.add_subplot(ga[2 * i + k, j])
            ax.imshow(im, cmap='gray' if j == 0 else 'magma', vmin=None if j == 0 else lo, vmax=None if j == 0 else hi); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0 and k == 0: ax.set_title(cols[j], fontsize=9, color=INK)
            if j == 0: ax.set_ylabel(f'{lab} {"bright" if k == 0 else "dim"}\nch {ch}', fontsize=8, color=INK, rotation=0, ha='right', va='center')
fig.text(.075, .885, 'a', fontsize=13, weight='bold', color=INK)
S = E['sets'][a.set]; rows = []
for st, v in S.items():
    for lab, e in v.items():
        if isinstance(e, dict) and 'r_realref' in e and 'r_avgread' in e: rows.append((lab.replace('single:', '').replace('grp:', 'all: '), e))
rows.sort(key=lambda t: -t[1]['r_realref'])
ax = fig.add_subplot(gs[1, 0]); y = np.arange(len(rows)); h = .26
for d, (key, col, name) in enumerate([('r_avgread', BLUE, 'flow (mean of samples)'), ('r_reg', ORANGE, 'regression'), ('r_shuffled', AQUA, 'shuffled-LF control')]):
    ax.barh(y + (d - 1) * h, [e.get(key, np.nan) for _, e in rows], h * .9, color=col, label=name)
ax.scatter([e['r_realref'] for _, e in rows], y, marker='|', s=160, color=GREY, label='real-image reference', zorder=3)
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows]); ax.invert_yaxis(); ax.axvline(0, color='#c8c8c4', lw=.8)
ax.set_xlabel(f'r with curated label ({"new blood draw" if a.set == "newdraw" else "held-out FOVs"})'); ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.45, -.13), ncol=2)
fig.text(.075, .385, 'b', fontsize=13, weight='bold', color=INK)
ax = fig.add_subplot(gs[1, 1]); mets = [('w1', 'marginal W1'), ('condw1', 'conditional W1'), ('crps', 'CRPS')]
x = np.arange(len(mets)); w = .26
for d, (arm, col, name) in enumerate([('sample', BLUE, 'flow'), ('reg', ORANGE, 'regression'), ('shuffled', AQUA, 'shuffled-LF control')]):
    vals = []
    for mk, _ in mets:
        k = f'{mk}_{arm}' if not (mk == 'crps' and arm == 'shuffled') else None
        vals.append(np.nanmean([e.get(k, np.nan) for _, e in rows]) if k else np.nan)
    ax.bar(x + (d - 1) * w, vals, w * .9, color=col, label=name)
    for xi, v in zip(x + (d - 1) * w, vals):
        if np.isfinite(v): ax.text(xi, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in mets]); ax.set_ylabel('distance to real readouts (sd units, lower = better)')
ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.07), ncol=3)
fig.text(.505, .385, 'c', fontsize=13, weight='bold', color=INK)
fig.savefig(a.out, dpi=150, bbox_inches='tight'); fig.savefig(a.out.replace('.png', '.pdf'), bbox_inches='tight'); print(a.out)
