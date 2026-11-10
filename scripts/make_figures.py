"""Overview figures for BSCCM (deliverables): fig1 = architecture (data -> models -> outputs, real cell thumbnails),
fig2 = capabilities (results panels from recorded result files). Usage: make_figures.py --thumbs THUMBS.npz --out DIR"""
import argparse, json, csv, glob, collections, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ap = argparse.ArgumentParser(); ap.add_argument('--thumbs', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
RES = '/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/'
# reference palette (light): blue, orange, aqua, yellow, magenta, green, violet, red; surface + ink
BLUE, ORANGE, AQUA, GREEN, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#008300', '#6250d6'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})
def tint(hexc, f):   # mix colour with white (f = colour share)
    c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255.; return tuple(1 - f * (1 - c))
TH = np.load(a.thumbs)

# ---------------------------------------------------------------- Figure 1: architecture
fig = plt.figure(figsize=(15, 8.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15); ax.set_ylim(0, 8.2); ax.axis('off')
def box(x0, y0, x1, y1, fc, text='', tc='white', fs=10, bold=True, r=0.12):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec='none', zorder=2))
    if text: ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', color=tc, fontsize=fs, fontweight='bold' if bold else 'normal', zorder=3, linespacing=1.25)
def arrow(pts, color=INK2, lw=1.8, dashed=False, both=False):
    xs, ys = zip(*pts); ls = (0, (4, 3)) if dashed else '-'
    if len(pts) > 2: ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle='butt', zorder=1)
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=1, arrowprops=dict(arrowstyle='<|-|>' if both else '-|>', color=color, lw=lw, ls=ls,
                shrinkA=0, shrinkB=0, mutation_scale=14))
def thumbs(imgs, x0, y0, size, gap=0.08, cmap='gray'):
    for i, im in enumerate(imgs):
        e = [x0 + i * (size + gap), x0 + i * (size + gap) + size, y0, y0 + size]
        ax.imshow(im, extent=e, cmap=cmap if im.ndim == 2 else None, interpolation='lanczos', zorder=3, aspect='auto')
def label(x, y, t, c=INK2, fs=9, ha='center', w='normal'): ax.text(x, y, t, ha=ha, va='center', color=c, fontsize=fs, fontweight=w, zorder=4)

# lanes (row centres)
YA, YB, YC = 6.35, 3.85, 1.35
# inputs (left)
box(0.25, YA - 0.95, 3.45, YA + 0.95, tint(BLUE, .16)); thumbs([TH['gran_bf'], TH['gran_df'], TH['gran_dpc']], 0.42, YA - 0.6, 0.92)
label(1.85, YA + 0.72, 'Label-free  ·  23 LEDs + phase', BLUE, 9, w='bold')
for i, t in enumerate(['bright', 'dark', 'phase']): label(0.88 + i * 1.0, YA - 0.78, t, INK2, 8)
box(0.25, YB - 0.95, 3.45, YB + 0.95, tint(VIOLET, .16)); thumbs([TH['gran_gs'], TH['lym_gs'], TH['mono_gs']], 0.42, YB - 0.6, 0.92)
label(1.85, YB + 0.72, 'Giemsa (Wright)  ·  same cells', VIOLET, 9, w='bold')
box(0.25, YC - 0.95, 3.45, YC + 0.95, tint(GREEN, .16)); thumbs([TH['mll_neutrophil_segmented'], TH['mll_eosinophil'], TH['mll_lymphocyte_large_granular']], 0.42, YC - 0.6, 0.92)
label(1.85, YC + 0.72, 'Clinical smears  ·  PBC, MLL23', GREEN, 9, w='bold')
# encoders
box(4.2, YA - 0.6, 6.3, YA + 0.6, BLUE, 'LF encoder\nResNet')
box(4.2, YC - 0.6, 6.3, YB + 0.6, VIOLET, 'Giemsa\nencoder\nDinoBloom')
# heads
box(7.1, YA - 0.6, 9.2, YA + 0.6, tint(BLUE, .22))
box(7.3, YA + 0.05, 9.0, YA + 0.45, ORANGE, 'markers', fs=9); box(7.3, YA - 0.45, 9.0, YA - 0.05, AQUA, 'autofluor.', fs=9)
box(7.1, YB - 0.6, 9.2, YB + 0.6, tint(VIOLET, .22))
box(7.3, YB + 0.05, 9.0, YB + 0.45, ORANGE, 'markers', fs=9); box(7.3, YB - 0.45, 9.0, YB - 0.05, AQUA, 'autofluor.', fs=9)
# supervision
box(7.1, 7.45, 9.2, 8.0, tint(ORANGE, .28), 'fluorescence · 412k cells', INK, 8.5, False)
arrow([(8.15, 7.45), (8.15, YA + 0.6)], ORANGE)
box(7.1, YC - 0.85, 9.2, YC - 0.3, tint(GREEN, .28), 'lineage anchors', INK, 8.5, False)
arrow([(7.1, YC - 0.575), (6.3, YC - 0.575)], GREEN)
# outputs
box(10.0, YA - 0.6, 12.1, YA + 0.6, ORANGE, 'markers from\nlabel-free')
box(10.0, YB - 0.6, 12.1, YB + 0.6, VIOLET, 'markers from\nGiemsa')
box(10.0, YC - 0.6, 12.1, YC + 0.6, GREEN, 'clinical cell\nprofiles')
# flow arrows
for y in (YA, YB, YC): arrow([(3.45, y), (4.2, y)])
arrow([(6.3, YA), (7.1, YA)]); arrow([(9.2, YA), (10.0, YA)])
arrow([(6.3, YB), (7.1, YB)]); arrow([(9.2, YB), (10.0, YB)])
arrow([(6.3, YC), (10.0, YC)])
# teacher -> student links (15.7k same-cell pairs)
arrow([(8.15, YA - 0.6), (8.15, YB + 0.6)], ORANGE); label(8.3, (YA + YB) / 2, 'distil', ORANGE, 9, 'left', 'bold')
arrow([(5.25, YA - 0.6), (5.25, YB + 0.6)], INK2, dashed=True, both=True); label(5.4, (YA + YB) / 2, 'align', INK2, 9, 'left', 'bold')
label(6.7, (YA + YB) / 2 - 0.32, '15.7k paired cells', MUTED, 8)
# evaluation (right)
box(12.9, YA - 0.6, 14.75, YA + 0.6, tint(ORANGE, .28), 'new blood\nisolation\nr = .64', INK, 8.5)
box(12.9, YB - 0.6, 14.75, YB + 0.6, tint(AQUA, .32), 'measured\nautofluor.\nr = .58 / .86', INK, 8.5)
box(12.9, YC - 0.6, 14.75, YC + 0.6, tint(GREEN, .28), 'cell-type\ncontrasts\nAUROC', INK, 8.5)
for y in (YA, YB, YC): arrow([(12.1, y), (12.9, y)], MUTED, 1.4, dashed=True)
label(13.825, 7.72, 'tested on', MUTED, 9, w='bold')
fig.savefig(a.out + '/fig1_architecture.png', dpi=220); fig.savefig(a.out + '/fig1_architecture.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: capabilities
SHORT = lambda n: n.replace('single:', '').replace('grp:CD3/CD19/CD56', 'CD3/19/56 grp').replace('grp:CD45', 'CD45 grp').replace('grp:CD16', 'CD16 grp')
fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), gridspec_kw=dict(wspace=0.55, hspace=0.5, left=0.07, right=0.98, top=0.93, bottom=0.08))
def clean(axx, grid_axis='x'):
    for s in ('top', 'right'): axx.spines[s].set_visible(False)
    axx.grid(axis=grid_axis, color=GRID, lw=0.8); axx.set_axisbelow(True)
# (a) LF -> markers on a new blood isolation (v5 labels, SSL-init r50, 2 seeds x 2 directions)
rs = [r for r in json.load(open(glob.glob(RES + 'N023-*/goal_runs_v5.json')[0])) if r['run'].startswith('n023_v5')]
mk = [k for k in rs[0] if k.startswith('r/') and 'CD123/HLA' not in k]
vals = {k: np.nanmean([r.get(k, np.nan) for r in rs]) for k in mk}; order = sorted(vals, key=vals.get)
A = axs[0, 0]; y = np.arange(len(order)); A.barh(y, [vals[k] for k in order], color=BLUE, height=0.62)
A.set_yticks(y, [SHORT(k[2:]) for k in order]); A.set_xlim(0, 1); A.set_xlabel('Pearson r'); clean(A)
A.axvline(np.mean(list(vals.values())), color=INK2, lw=1, ls=(0, (3, 2))); A.text(np.mean(list(vals.values())) + 0.02, len(order) - 0.4, f'mean {np.mean(list(vals.values())):.2f}', color=INK2, fontsize=8)
A.set_title('a  Label-free → markers, new blood isolation')
# (b) LED information map
rows = list(csv.DictReader(open('/home/hpc/v181bb/v181bb10/bsccm_deliverables/results/lf_information_map_fov_v4.csv')))
cols = [('bf', 'bright'), ('phase', 'phase'), ('dpc4', 'DPC'), ('DF_low', 'dark lo'), ('DF_high', 'dark hi')]
rows = [r for r in rows if 'CD123/HLA' not in r['marker'] and r['marker'] not in ('autofl', 'r_mean')]
rows = sorted(rows, key=lambda r: -float(r['ALL']))
M = np.array([[float(r[c]) / float(r['ALL']) for c, _ in cols] for r in rows]); B = axs[0, 1]   # share of the all-LED signal
im = B.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b', [tint(BLUE, .06), BLUE, '#123f78']), vmin=0.5, vmax=1.0, aspect='auto')
B.set_xticks(range(len(cols)), [t for _, t in cols], rotation=0); B.set_yticks(range(len(rows)), [SHORT(r['marker']) for r in rows])
for s_ in B.spines.values(): s_.set_visible(False)
B.tick_params(length=0); cb = fig.colorbar(im, ax=B, fraction=0.05, pad=0.03); cb.outline.set_visible(False); cb.set_label('share of all-LED r', color=INK2)
B.set_title('b  Signal per illumination (share of all LEDs)')
# (c) Giemsa -> same-cell autofluorescence truth (within-slide r)
met = list(csv.DictReader(open(glob.glob(RES + 'N025-*/n025_metrics.csv')[0])))
C = axs[0, 2]; arms = [('v4r18', 'one teacher', MUTED), ('tt6', 'two teachers', VIOLET)]
for i, (sl, nm) in enumerate([('gs_autofl_within_b0', 'slide 1'), ('gs_autofl_within_b1', 'slide 2')]):
    for j, (arm, lab, col) in enumerate(arms):
        v = [float(r['value']) for r in met if r['metric'] == sl and r['arm'] == arm]; x = i + (j - 0.5) * 0.36
        C.bar(x, np.mean(v), width=0.32, color=col, label=lab if i == 0 else None); C.scatter([x] * len(v), v, s=14, color=INK, zorder=3, lw=0)
C.set_xticks([0, 1], ['slide 1', 'slide 2']); C.set_ylim(0, 1); C.set_ylabel('Pearson r (within slide)'); clean(C, 'y'); C.legend(frameon=False, fontsize=8, loc='upper left')
C.set_title('c  Giemsa → measured autofluorescence')
# (d) Giemsa -> markers, transitive estimate (two-teacher chain, mean of both slide directions)
est = [r for r in csv.DictReader(open(glob.glob(RES + 'N025-*/gs_mk_estimates_v4_v5_tt6.csv')[0])) if r['chain'] == 'tt6' and r['marker'].startswith(('single', 'grp')) and 'CD123/HLA' not in r['marker']]
ev = collections.defaultdict(list)
for r in est: ev[r['marker']].append(float(r['r_est']))
eo = sorted(ev, key=lambda k: np.mean(ev[k])); D_ = axs[1, 0]; y = np.arange(len(eo))
D_.barh(y, [np.mean(ev[k]) for k in eo], color=tint(VIOLET, .75), height=0.62)
for i, k in enumerate(eo): D_.scatter(ev[k], [i] * len(ev[k]), s=12, color=INK, zorder=3, lw=0)
D_.set_yticks(y, [SHORT(k) for k in eo]); D_.set_xlim(0, 1); D_.set_xlabel('estimated r (via label-free)'); clean(D_)
D_.set_title('d  Giemsa → markers (transitive estimate)')
# (e) clinical MLL23 pre-registered contrasts (two-teacher student, 3 seeds)
tr = list(csv.DictReader(open(glob.glob(RES + 'N025-*/transfer_multiseed.csv')[0])))
def contrast_panel(axx, dset, title):
    cs = collections.defaultdict(list); base = collections.defaultdict(list)
    for r in tr:
        if r['set'] != dset: continue
        (cs if r['arm'] == 'tt6' else base if r['arm'] == 'v4r18' else {}).setdefault(r['contrast'], []).append(float(r['auroc'])) if r['arm'] in ('tt6', 'v4r18') else None
    order = sorted(cs, key=lambda k: np.mean(cs[k])); yy = np.arange(len(order))
    axx.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 2)))
    for i, k in enumerate(order):
        axx.plot([min(cs[k]), max(cs[k])], [i, i], color=tint(VIOLET, .45), lw=3, solid_capstyle='round', zorder=2)
        axx.scatter([np.mean(cs[k])], [i], s=42, color=VIOLET, zorder=3, edgecolor=SURF, lw=1.2)
    axx.set_yticks(yy, [k.replace('(held-out)', '').replace('neutrophil', 'neut').replace('lymphocyte', 'lym').replace('monocyte', 'mono').replace('grpCD3/19/56', 'CD3/19/56').replace('erythroblast', 'erythrobl.').replace('basophil', 'baso').replace('eosinophil', 'eos').strip() for k in order], fontsize=8)
    axx.set_xlim(0.2, 1.02); axx.set_xlabel('AUROC  (0.5 = chance)'); clean(axx); axx.set_title(title)
contrast_panel(axs[1, 1], 'MLL23', 'e  Clinical cells, MLL23 (never trained on)')
contrast_panel(axs[1, 2], 'PBC-B', 'f  Clinical cells, PBC held-out half')
fig.savefig(a.out + '/fig2_capabilities.png', dpi=220); fig.savefig(a.out + '/fig2_capabilities.pdf'); plt.close(fig)
print('wrote', a.out + '/fig1_architecture.{png,pdf}', a.out + '/fig2_capabilities.{png,pdf}')

# ---------------------------------------------------------------- Figure 3: virtual Wright staining (held-out cells)
def fig_virtual_stain(run_dir, out, n=6, seed=4):
    import sys as _s; _s.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM/scripts')
    from vs_common import registered_rgb
    W = '/hnvme/workspace/v181bb10-bsccm'; G = np.load(run_dir + '/eval/generated.npz'); R = np.load(run_dir + '/register.npz')
    H = np.load(W + '/pack/histology.npy', mmap_mode='r'); L = np.load(W + '/pack/led_array.npy', mmap_mode='r'); rs = float(G['rgb_scale'])
    D = json.load(open(run_dir + '/downstream/downstream.json')) if os.path.exists(run_dir + '/downstream/downstream.json') else json.load(open(run_dir + '/downstream_tt6s0/downstream.json'))
    idx = np.random.default_rng(seed).choice(len(G['hist_row']), n, replace=False)
    fig = plt.figure(figsize=(15, 6.2)); gs = fig.add_gridspec(3, n + 2, width_ratios=[1] * n + [0.95, 2.6], hspace=0.08, wspace=0.06, left=0.06, right=0.98, top=0.9, bottom=0.1)
    rows = [('label-free', BLUE), ('virtual stain', VIOLET), ('real stain', INK2)]
    for c, i in enumerate(idx):
        r = int(G['hist_row'][i]); g = int(G['gidx'][i]); bf = np.array(L[g]).view(np.int16)[0].astype(np.float32); lo, hi = np.percentile(bf, [1, 99.5])
        real = np.clip(registered_rgb(np.asarray(H[r], np.float32), R['scale'][r], R['dy'][r], R['dx'][r], 128) / rs, 0, 1)
        for k, im in enumerate([np.clip((bf - lo) / (hi - lo), 0, 1), G['images'][i] / 255., real]):
            ax = fig.add_subplot(gs[k, c]); ax.imshow(im, cmap='gray' if im.ndim == 2 else None); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values(): s_.set_visible(False)
            if c == 0: ax.set_ylabel(rows[k][0], color=rows[k][1], fontsize=10, fontweight='bold')
    ax = fig.add_subplot(gs[:, n + 1]); ag = D['agreement']
    ks = [k for k in ag if k != 'autofl' and 'CD123/HLA' not in k]; ks = sorted(ks, key=lambda k: ag[k]['gen_vs_reg'])
    y = np.arange(len(ks)); ax.barh(y, [ag[k]['gen_vs_reg'] for k in ks], color=VIOLET, height=0.62)
    ax.set_yticks(y, [k.replace('single:', '').replace('grp:CD3/CD19/CD56', 'CD3/19/56 grp').replace('grp:CD45', 'CD45 grp').replace('grp:CD16', 'CD16 grp') for k in ks])
    ax.set_xlim(0, 1); ax.set_xlabel('agreement r, virtual vs real'); ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    ax.set_title('Giemsa student reads the virtual stain', loc='left')
    fig.suptitle('Virtual Wright staining from label-free LED images  ·  cells from a slide the generator never saw', x=0.06, ha='left', fontsize=11, fontweight='bold')
    fig.savefig(out + '/fig3_virtual_staining.png', dpi=220); fig.savefig(out + '/fig3_virtual_staining.pdf'); plt.close(fig)
if os.environ.get('VS_RUN'): fig_virtual_stain(os.environ['VS_RUN'], a.out); print('wrote fig3')
