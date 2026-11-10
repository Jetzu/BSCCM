"""Deliverable figures, state 2026-11-10 (N031-N033): fig1 = final architecture (two-scale marker teacher, autofluorescence teacher,
Giemsa student, clinical path, virtual staining), fig2 = capabilities (8 panels from recorded result files).
Usage: make_figures_v2.py --thumbs THUMBS.npz --out DIR"""
import argparse, json, csv, glob, collections, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ap = argparse.ArgumentParser(); ap.add_argument('--thumbs', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
RES = '/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/'; W = '/hnvme/workspace/v181bb10-bsccm'
BLUE, ORANGE, AQUA, GREEN, VIOLET, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#008300', '#6250d6', '#e87ba4'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def tint(h, f): c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.; return tuple(1 - f * (1 - c))
TH = np.load(a.thumbs)
SHORT = lambda n: n.replace('single:', '').replace('grp:CD3/CD19/CD56', 'CD3/19/56 grp').replace('grp:CD45', 'CD45 grp').replace('grp:CD16', 'CD16 grp')

# ================================================================ Figure 1: architecture
fig = plt.figure(figsize=(16, 8.95)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16); ax.set_ylim(0.9, 9.85); ax.axis('off')
def box(x0, y0, x1, y1, fc, text='', tc='white', fs=10, bold=True):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.12', fc=fc, ec='none', zorder=2))
    if text: ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', color=tc, fontsize=fs, fontweight='bold' if bold else 'normal', zorder=3, linespacing=1.25)
def arrow(pts, color=INK2, lw=1.8, dashed=False, both=False):
    xs, ys = zip(*pts); ls = (0, (4, 3)) if dashed else '-'
    if len(pts) > 2: ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=1)
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=1, arrowprops=dict(arrowstyle='<|-|>' if both else '-|>', color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=14))
def thumbs(imgs, x0, y0, s, gap=0.08):
    for i, im in enumerate(imgs):
        ax.imshow(im, extent=[x0 + i * (s + gap), x0 + i * (s + gap) + s, y0, y0 + s], cmap='gray' if im.ndim == 2 else None, interpolation='lanczos', zorder=3, aspect='auto')
def label(x, y, t, c=INK2, fs=9, ha='center', w='normal'): ax.text(x, y, t, ha=ha, va='center', color=c, fontsize=fs, fontweight=w, zorder=4)
Y0, Y1, Y2, Y3, Y4 = 9.2, 7.35, 6.2, 4.45, 2.3            # lanes: virtual stain, marker teacher, autofl teacher, Giemsa, clinical
XI0, XI1, XE0, XE1, XH0, XH1, XO0, XO1, XT0, XT1 = 0.25, 3.45, 4.25, 6.65, 7.45, 9.25, 10.05, 12.05, 12.9, 15.75
# inputs
box(XI0, 5.55, XI1, 7.95, tint(BLUE, .16)); thumbs([TH['gran_bf'], TH['gran_df'], TH['gran_dpc']], 0.42, 6.05, 0.92)
label(1.85, 7.68, 'Label-free  ·  23 LEDs + phase', BLUE, 9, w='bold')
for i, t in enumerate(['bright', 'dark', 'phase']): label(0.88 + i * 1.0, 5.85, t, INK2, 8)
box(XI0, Y3 - 0.95, XI1, Y3 + 0.95, tint(VIOLET, .16)); thumbs([TH['gran_gs'], TH['lym_gs'], TH['mono_gs']], 0.42, Y3 - 0.66, 0.92)
label(1.85, Y3 + 0.72, 'Giemsa (Wright)  ·  same cells', VIOLET, 9, w='bold')
box(XI0, Y4 - 0.95, XI1, Y4 + 0.95, tint(GREEN, .16)); thumbs([TH['mll_neutrophil_segmented'], TH['mll_eosinophil'], TH['mll_lymphocyte_large_granular']], 0.42, Y4 - 0.66, 0.92)
label(1.85, Y4 + 0.72, 'Clinical smears  ·  PBC, MLL23', GREEN, 9, w='bold')
# virtual staining lane
box(XE0, Y0 - 0.35, XE1, Y0 + 0.35, MAGENTA, 'virtual stainer  ·  UNet')
box(XO0, Y0 - 0.45, XO1, Y0 + 0.45, tint(MAGENTA, .28)); ax.imshow(TH['vs_gen'], extent=[XO0 + 0.12, XO0 + 0.86, Y0 - 0.37, Y0 + 0.37], interpolation='lanczos', zorder=3, aspect='auto')
label((XO0 + 0.95 + XO1) / 2, Y0, 'virtual\nGiemsa', INK, 9, w='bold')
arrow([(1.85, 7.95), (1.85, Y0), (XE0, Y0)]); arrow([(XE1, Y0), (XO0, Y0)])
# teachers
box(XE0, Y1 - 0.38, XE1, Y1 + 0.38, BLUE, 'marker teacher\nResNet-50 · full + 2× zoom', fs=9)
box(XE0, Y2 - 0.38, XE1, Y2 + 0.38, AQUA, 'autofluor. teacher\nResNet-18 · unstained slides', INK, 9)
arrow([(XI1, Y1), (XE0, Y1)]); arrow([(XI1, Y2), (XE0, Y2)])
box(XH0, 5.7, XH1, 7.85, tint(BLUE, .22)); box(XH0 + 0.2, Y1 - 0.25, XH1 - 0.2, Y1 + 0.25, ORANGE, 'markers', fs=9); box(XH0 + 0.2, Y2 - 0.25, XH1 - 0.2, Y2 + 0.25, AQUA, 'autofluor.', INK, 9)
arrow([(XE1, Y1), (XH0 + 0.2, Y1)]); arrow([(XE1, Y2), (XH0 + 0.2, Y2)])
box(XH0, 8.15, XH1, 8.65, tint(ORANGE, .3), 'fluorescence · 412k cells', INK, 8.5, False)
arrow([((XH0 + XH1) / 2, 8.15), ((XH0 + XH1) / 2, 7.85)], ORANGE)
box(XO0, Y1 - 0.4, XO1, Y1 + 0.4, ORANGE, 'markers from\nlabel-free')
arrow([(XH1 - 0.2, Y1), (XO0, Y1)])
# Giemsa student
box(XE0, Y4 - 0.6, XE1, Y3 + 0.6, VIOLET, 'Giemsa student\nDinoBloom-B\n4 blocks trained')
box(XH0, Y3 - 0.6, XH1, Y3 + 0.6, tint(VIOLET, .22)); box(XH0 + 0.2, Y3 + 0.05, XH1 - 0.2, Y3 + 0.45, ORANGE, 'markers', fs=9); box(XH0 + 0.2, Y3 - 0.45, XH1 - 0.2, Y3 - 0.05, AQUA, 'autofluor.', INK, 9)
for y in (Y3, Y4): arrow([(XI1, y), (XE0, y)])
arrow([(XE1, Y3), (XH0, Y3)])
arrow([((XH0 + XH1) / 2, 5.7), ((XH0 + XH1) / 2, Y3 + 0.6)], ORANGE); label((XH0 + XH1) / 2 + 0.15, 5.18, 'distil', ORANGE, 9, 'left', 'bold')
arrow([((XE0 + XE1) / 2, Y2 - 0.38), ((XE0 + XE1) / 2, Y3 + 0.6)], INK2, dashed=True, both=True); label((XE0 + XE1) / 2 + 0.15, 5.18, 'align', INK2, 9, 'left', 'bold')
label((XE1 + XH0) / 2, 4.95, '15.7k\npairs', MUTED, 8)
box(XO0, Y3 - 0.4, XO1, Y3 + 0.4, VIOLET, 'markers from\nGiemsa'); arrow([(XH1, Y3), (XO0, Y3)])
box(XH0, Y4 - 0.95, XH1, Y4 - 0.45, tint(GREEN, .3), 'lineage anchors', INK, 8.5, False); arrow([(XH0, Y4 - 0.7), (XE1, Y4 - 0.7)], GREEN)
box(XO0, Y4 - 0.4, XO1, Y4 + 0.4, GREEN, 'clinical cell\nprofiles'); arrow([(XE1, Y4 + 0.1), (XO0, Y4 + 0.1)])
# tested on
for y, t, fc in [(Y0, 'cell-specific\nretrieval 34-84× chance', MAGENTA), (Y1, 'new blood isolation\nr = .65', ORANGE),
                 (Y3, 'measured autofluor.\nr = .56 / .85', AQUA), (Y4, 'lineage AUROC\n.87-1.00', GREEN)]:
    box(XT0, y - 0.4, XT1, y + 0.4, tint(fc, .3), t, INK, 8.5); arrow([(XO1, y), (XT0, y)], MUTED, 1.4, dashed=True)
label((XT0 + XT1) / 2, 8.5, 'tested on', MUTED, 9, w='bold')
fig.savefig(a.out + '/fig1_architecture.png', dpi=220); fig.savefig(a.out + '/fig1_architecture.pdf'); plt.close(fig)

# ================================================================ Figure 2: capabilities
fig, axs = plt.subplots(2, 4, figsize=(20, 9.2), gridspec_kw=dict(wspace=0.62, hspace=0.5, left=0.055, right=0.99, top=0.93, bottom=0.08))
def clean(axx, g='x'):
    for s_ in ('top', 'right'): axx.spines[s_].set_visible(False)
    axx.grid(axis=g, color=GRID, lw=0.8); axx.set_axisbelow(True)
T5 = np.load(W + '/targets_v5.npz', allow_pickle=True); Y5, M5 = T5['Y'], T5['mask']; names = [str(n) for n in T5['names']]
c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
def ens(runs):
    Ps = [np.load(f'{W}/runs/{r}/eval_preds.npz') for r in runs]; G = Ps[0]['G']; P = np.mean([p['P'] for p in Ps], 0)
    return {j: np.corrcoef(P[M5[G, j], j], Y5[G[M5[G, j]], j])[0, 1] for j in c11}
rb10, rb01 = ens(['n031_base_b10', 'n031_zoom2_b10']), ens(['n031_base_b01', 'n031_zoom2_b01'])
vals = {names[j]: (rb10[j] + rb01[j]) / 2 for j in c11}; order = sorted(vals, key=vals.get); A_ = axs[0, 0]; y = np.arange(len(order))
A_.barh(y, [vals[k] for k in order], color=BLUE, height=0.62); A_.set_yticks(y, [SHORT(k) for k in order]); A_.set_xlim(0, 1); A_.set_xlabel('Pearson r'); clean(A_)
mu = np.mean(list(vals.values())); A_.axvline(mu, color=INK2, lw=1, ls=(0, (3, 2))); A_.text(mu + 0.02, len(order) - 0.4, f'mean {mu:.2f}', color=INK2, fontsize=8)
A_.set_title('a  Label-free → markers, new blood isolation')
rows = [r for r in csv.DictReader(open('/home/hpc/v181bb/v181bb10/bsccm_deliverables/results/lf_information_map_fov_v4.csv')) if 'CD123/HLA' not in r['marker'] and r['marker'] not in ('autofl', 'r_mean')]
rows = sorted(rows, key=lambda r: -float(r['ALL'])); cols = [('bf', 'bright'), ('phase', 'phase'), ('dpc4', 'DPC'), ('DF_low', 'dark lo'), ('DF_high', 'dark hi')]
Mx = np.array([[float(r[c]) / float(r['ALL']) for c, _ in cols] for r in rows]); B_ = axs[0, 1]
im = B_.imshow(Mx, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', [tint(BLUE, .06), BLUE, '#123f78']), vmin=0.5, vmax=1, aspect='auto')
B_.set_xticks(range(len(cols)), [t for _, t in cols]); B_.set_yticks(range(len(rows)), [SHORT(r['marker']) for r in rows])
for s_ in B_.spines.values(): s_.set_visible(False)
B_.tick_params(length=0); cb = fig.colorbar(im, ax=B_, fraction=0.05, pad=0.03); cb.outline.set_visible(False); cb.set_label('share of all-LED r', color=INK2)
B_.set_title('b  Signal per illumination')
C_ = axs[0, 2]; steps = [('ResNet-18', .6119), ('SSL\nResNet-50', .6343), ('repaired\nlabels', .6404), ('multi-scale\nensemble', .6515), ('inductive\npretraining', .6527)]
C_.plot(range(len(steps)), [v for _, v in steps], color=BLUE, lw=2, marker='o', ms=8, mfc=BLUE, mec=SURF, mew=1.5)
for i, (_, v) in enumerate(steps): C_.text(i, v + 0.0035, f'{v:.3f}', ha='center', color=INK2, fontsize=8)
C_.set_xticks(range(len(steps)), [s for s, _ in steps], fontsize=8); C_.set_ylim(0.603, 0.665); C_.set_ylabel('GOAL r (11 markers, new isolation)'); clean(C_, 'y')
C_.set_title('c  Progress on the GOAL')
D_ = axs[0, 3]; ag = json.load(open(W + '/runs/n028_vs_batch0/downstream/downstream.json'))['agreement']
ks = sorted([k for k in ag if k != 'autofl' and 'CD123/HLA' not in k], key=lambda k: ag[k]['gen_vs_reg']); y = np.arange(len(ks))
D_.barh(y, [ag[k]['gen_vs_reg'] for k in ks], color=MAGENTA, height=0.62); D_.set_yticks(y, [SHORT(k) for k in ks]); D_.set_xlim(0, 1); D_.set_xlabel('agreement r, virtual vs real stain'); clean(D_)
D_.set_title('d  Giemsa student on virtual stains')
met = list(csv.DictReader(open(RES + 'N029-clean-calibration-cohort-no-giemsa-hist/n029_metrics.csv'))) + list(csv.DictReader(open(RES + 'N032-teacher-run-variance-of-clinical-transfe/n032_n033_metrics.csv')))
E_ = axs[1, 0]; groups = [('one teacher', MUTED, lambda r: r['arm'] == 'v4c'), ('two teachers', VIOLET, lambda r: r['arm'] in ('tt6c', 't0', 't1', 't2'))]
for i, (sl, nm) in enumerate([('gs_autofl_within_b0', 'slide 1'), ('gs_autofl_within_b1', 'slide 2')]):
    for j, (lab, col, f) in enumerate(groups):
        v = sorted({(r['run'], float(r['value'])) for r in met if r['metric'] == sl and f(r)}); v = [x for _, x in v]; x = i + (j - 0.5) * 0.36
        E_.bar(x, np.mean(v), width=0.32, color=col, label=lab if i == 0 else None); E_.scatter([x] * len(v), v, s=12, color=INK, zorder=3, lw=0)
E_.set_xticks([0, 1], ['slide 1', 'slide 2']); E_.set_ylim(0, 1); E_.set_ylabel('Pearson r within slide'); clean(E_, 'y'); E_.legend(frameon=False, fontsize=8, loc='upper left')
E_.set_title('e  Giemsa → measured autofluorescence')
est = [r for r in csv.DictReader(open(RES + 'N025-giemsa-chain-without-the-trade-off-two-t/gs_mk_estimates_v4_v5_tt6.csv')) if r['chain'] == 'tt6' and r['marker'].startswith(('single', 'grp')) and 'CD123/HLA' not in r['marker']]
ev = collections.defaultdict(list)
for r in est: ev[r['marker']].append(float(r['r_est']))
eo = sorted(ev, key=lambda k: np.mean(ev[k])); F_ = axs[1, 1]; y = np.arange(len(eo))
F_.barh(y, [np.mean(ev[k]) for k in eo], color=tint(VIOLET, .75), height=0.62)
for i, k in enumerate(eo): F_.scatter(ev[k], [i] * len(ev[k]), s=12, color=INK, zorder=3, lw=0)
F_.set_yticks(y, [SHORT(k) for k in eo]); F_.set_xlim(0, 1); F_.set_xlabel('estimated r (via label-free)'); clean(F_)
F_.set_title('f  Giemsa → markers (transitive estimate)')
tr = list(csv.DictReader(open(RES + 'N032-teacher-run-variance-of-clinical-transfe/transfer_all.csv')))
def contrast(axx, dset, title):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in tr:
        if r['set'] == dset and r['arm'] in ('t0', 't1', 't2', 'old_v4teacher'): per[r['contrast']][r['arm']].append(float(r['auroc']))
    tm = {c: [np.mean(v) for v in d.values()] for c, d in per.items()}; order = sorted(tm, key=lambda c: np.mean(tm[c])); yy = np.arange(len(order))
    axx.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 2)))
    for i, c in enumerate(order):
        axx.plot([min(tm[c]), max(tm[c])], [i, i], color=tint(GREEN, .45), lw=3, solid_capstyle='round', zorder=2)
        axx.scatter([np.mean(tm[c])], [i], s=42, color=GREEN, zorder=3, edgecolor=SURF, lw=1.2)
    nice = lambda k: k.replace('(held-out)', '').replace('neutrophil', 'neut').replace('lymphocyte', 'lym').replace('monocyte', 'mono').replace('grpCD3/19/56', 'CD3/19/56').replace('erythroblast', 'erythrobl.').replace('basophil', 'baso').replace('eosinophil', 'eos').strip()
    axx.set_yticks(yy, [nice(k) for k in order], fontsize=8); axx.set_xlim(0.2, 1.02); axx.set_xlabel('AUROC (range over 4 teacher runs)'); clean(axx); axx.set_title(title)
contrast(axs[1, 2], 'MLL23', 'g  Clinical cells, MLL23')
contrast(axs[1, 3], 'PBC-B', 'h  Clinical cells, PBC held-out half')
fig.savefig(a.out + '/fig2_capabilities.png', dpi=200); fig.savefig(a.out + '/fig2_capabilities.pdf'); plt.close(fig)
print('wrote fig1/fig2')
