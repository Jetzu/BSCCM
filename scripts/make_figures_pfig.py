"""BSCCM paper figures in the house style (pfig, ~/.claude/skills/paper-figures), state 2026-11-17 (N013-N042).
Every number is read from saved predictions / result files; captions (tests, n, Holm p, sources) go to the .json sidecars.
  fig1 architecture (diagram)            fig2 label-free -> markers in a new blood draw (GOAL)
  fig3 labels limit the score            fig4 Giemsa student and clinical transfer
  fig5 marker images by flow matching    fig6 set-level distribution matching
Usage: make_figures_pfig.py [fig1 fig2 ...]   (default: all) -> ~/BSCCM/figures/<sub>/<name>.{png,svg,json}"""
import os, sys, json, csv, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
ROOT = '/home/hpc/v181bb/v181bb10/BSCCM'; RES = ROOT + '/.agent-research/results/'; W = '/hnvme/workspace/v181bb10-bsccm'
DELIV = '/home/hpc/v181bb/v181bb10/bsccm_deliverables'
SKY, GREY = pfig.PALETTE[0], pfig.BASELINE
T5 = np.load(W + '/targets_v5.npz', allow_pickle=True); Y5, M5, BT, FOV = T5['Y'], T5['mask'], T5['batch'], T5['fov']
NAMES = [str(n) for n in T5['names']]; C11 = [j for j, n in enumerate(NAMES[:12]) if 'CD123/HLA' not in n]
WEAK = ['single:CD19', 'single:CD56', 'single:CD123', 'single:HLA-DR']
def short(n): return (n.replace('single:', '').replace('grp:CD3/CD19/CD56', 'CD3/19/56 pool').replace('grp:CD45', 'CD45 pool')
                       .replace('grp:CD16', 'CD16 pool').replace('autofl', 'autofluor.'))
def load(run): E = np.load(f'{W}/runs/{run}/eval_preds.npz'); return E['P'], E['G']
def per_target(runs_by_dir):
    """{dir: [runs]} -> {(dir, target): r} on v5 labels, common-11, ensemble = mean of the runs' predictions."""
    out = {}
    for d, runs in runs_by_dir.items():
        Ps = [load(r) for r in runs]; G = Ps[0][1]; assert all(np.array_equal(p[1], G) for p in Ps); P = np.mean([p[0] for p in Ps], 0)
        for j in C11:
            m = M5[G, j]; out[(d, NAMES[j])] = float(np.corrcoef(P[m, j], Y5[G[m], j])[0, 1])
    return out
KEYS = [(d, NAMES[j]) for d in ('b10', 'b01') for j in C11]          # fixed order of the 22 target x direction items
vec = lambda dct: np.array([dct[k] for k in KEYS])


def fig2():
    """GOAL: label-free -> 11 surface markers in a new blood draw."""
    fig, ax = pfig.figure(2, 3, height_mm=120); ax = ax.ravel()
    # a milestones, 22 target x direction values each, paired vs the final model
    ms = [('ResNet-18', {'b10': ['n013_lf_sup_v4_b1to0_s0'], 'b01': ['n013_lf_sup_v4_b0to1_s0']}),
          ('SSL R50', {'b10': ['n014_sslinit_b1to0'], 'b01': ['n014_sslinit_b0to1']}),
          ('Clean labels', {'b10': ['n023_v5_b10'], 'b01': ['n023_v5_b01']}),
          ('Two scales', {'b10': ['n023_v5_b10', 'n026_zoom2_b10'], 'b01': ['n023_v5_b01', 'n026_zoom2_b01']}),
          ('Final', {'b10': ['n031_base_b10', 'n031_zoom2_b10'], 'b01': ['n031_base_b01', 'n031_zoom2_b01']})]
    vals = [vec(per_target(r)) for _, r in ms]; names = [n for n, _ in ms]
    pfig.bar(ax[0], vals, [GREY] * 4 + [SKY], names=names); ax[0].set_ylabel('Pearson r, new blood draw')
    ax[0].tick_params(axis='x', labelrotation=30); [t.set_ha('right') for t in ax[0].get_xticklabels()]
    pfig.stats_vs_ref(ax[0], 'Final', paired=True)
    # b per-marker r of the final model, two directions as rings
    fin = per_target(ms[-1][1]); order = sorted(C11, key=lambda j: np.mean([fin[(d, NAMES[j])] for d in ('b10', 'b01')]))
    pfig.bar(ax[1], [[fin[(d, NAMES[j])] for d in ('b10', 'b01')] for j in order], [SKY] * len(order), names=[short(NAMES[j]) for j in order])
    ax[1].set_ylabel('Pearson r, new blood draw'); ax[1].tick_params(axis='x', labelrotation=60); [t.set_ha('right') for t in ax[1].get_xticklabels()]
    pfig.no_stats(ax[1], 'descriptive per-marker profile; markers are not compared with each other')
    # c information per illumination group (FOV split, v4), share of the all-LED r
    rows = [r for r in csv.DictReader(open(DELIV + '/results/lf_information_map_fov_v4.csv')) if 'CD123/HLA' not in r['marker'] and r['marker'] not in ('autofl', 'r_mean')]
    rows = sorted(rows, key=lambda r: -float(r['ALL'])); cols = [('bf', 'Bright'), ('phase', 'Phase'), ('dpc4', 'DPC'), ('DF_low', 'Dark low'), ('DF_high', 'Dark high')]
    Mx = np.array([[float(r[c]) / float(r['ALL']) for c, _ in cols] for r in rows])
    pfig.heatmap(ax[2], Mx, [short(r['marker']) for r in rows], [t for _, t in cols], vmin=0.5, vmax=1.0, label='Share of all-LED r')
    # d label-count learning curve (N040): weak vs other markers
    lc = pd.read_csv(RES + 'N040-dense-label-completion-noisy-student-lab/n040_lc_per_target.csv')
    for grp, c, lab in ((WEAK, pfig.PALETTE[3], 'CD19, CD56, CD123, HLA-DR'), (None, SKY, 'Other 7 markers')):
        sel = lc[lc.target.isin(WEAK)] if grp else lc[~lc.target.isin(WEAK)]
        g = sel.groupby(['frac', 'dir']).r.mean().unstack()
        pfig.line(ax[3], g.index.values, g.mean(1).values, c, label=lab, err=g.std(1, ddof=1).values)
    ax[3].set_xlabel('Training cells (%)'); ax[3].set_ylabel('Pearson r, new blood draw'); ax[3].set_xticks([25, 50, 100]); ax[3].legend(loc='lower right')
    # e in-distribution reference vs held-out (N039), paired over the same 22 items
    ce = pd.read_csv(RES + 'N039-label-v8-re-derive-single-antibody-marke/n039_ceiling_per_target.csv')
    cev = lambda arm: np.array([ce[(ce.arm == arm) & (ce.draw == (0 if d == 'b10' else 1)) & (ce.target == t)].r.iloc[0] for d, t in KEYS])
    arms = [('Seen draw', 'fov_v5_full'), ('Seen, 128k', 'fov_v5_128k'), ('Final', 'goal_ens')]
    pfig.bar(ax[4], [cev(a) for _, a in arms], [pfig.PALETTE[1], pfig.PALETTE[2], SKY], names=[n for n, _ in arms])
    ax[4].set_ylabel('Pearson r, same test cells'); pfig.stats_vs_ref(ax[4], 'Final', paired=True)
    # f low-pass input probe (N042), paired per target x direction
    lp = pd.read_csv(RES + 'N042-weak-marker-architecture-low-pass-input/n042_lowpass_per_target.csv')
    a_ = np.array([lp[(lp.arm == 'base') & (lp.dir == d) & (lp.target == t)].r.iloc[0] for d, t in KEYS])
    b_ = np.array([lp[(lp.arm == 'lowpass1.5') & (lp.dir == d) & (lp.target == t)].r.iloc[0] for d, t in KEYS])
    pfig.paired(ax[5], a_, b_, [GREY, pfig.PALETTE[4]], ['Full detail', 'Blurred 1.5 px']); ax[5].set_ylabel('Pearson r, new blood draw')
    return pfig.save(fig, name='fig2_markers_new_draw', sub='paper', root=ROOT,
                     source=['runs/*/eval_preds.npz (n013, n014, n023, n026, n031, n039, n040, n042)', 'targets_v5.npz',
                             DELIV + '/results/lf_information_map_fov_v4.csv'],
                     notes='a: GOAL milestones, 22 = 11 markers x 2 directions per group, paired vs final; b: final model per marker, '
                           'rings = two directions; c: FOV split v4 information map; d: mean over directions, whiskers = sd over the 2 '
                           'directions; e: in-distribution FOV-split models (seen both draws; full 318k / data-matched 128k cells) vs held-out '
                           'final ensemble on identical cells; f: input low-pass sigma 1.5 px vs N031 base.')


def fig3():
    """Labels limit the score: FOV-level background-estimation error in the dim-dye targets (N036, N038, N039)."""
    fig, ax = pfig.figure(2, 3, height_mm=118); ax = ax.ravel()
    d2 = pd.read_csv(RES + 'N038-review-lf-mk-common-11-batch-held-out-go/d2_within_fov.csv')
    # a pooled vs within-FOV scoring of the same predictions
    pfig.paired(ax[0], d2.r_pooled.values, d2.r_within.values, [GREY, SKY], ['Pooled', 'Within FOV']); ax[0].set_ylabel('Pearson r, final model')
    # b FOV-level variance (excess over a permutation null): labels vs predictions
    pfig.paired(ax[1], d2.pred_fov_excess.values, d2.label_fov_excess.values, [SKY, pfig.PALETTE[3]], ['Prediction', 'Label'])
    ax[1].set_ylabel('FOV-level variance share')
    # c the more FOV-level label variance, the more within-FOV scoring recovers
    dim = d2.target.isin(['single:CD19', 'single:CD56', 'single:CD3'])
    pfig.scatter(ax[2], d2.label_fov_excess[~dim], (d2.r_within - d2.r_pooled)[~dim], SKY, label='Other markers')
    pfig.scatter(ax[2], d2.label_fov_excess[dim], (d2.r_within - d2.r_pooled)[dim], pfig.PALETTE[3], label='CD3, CD19, CD56')
    ax[2].set_xlabel('FOV-level label variance share'); ax[2].set_ylabel('Within-FOV minus pooled r'); ax[2].legend(loc='upper left')
    # d position correction fitted in-sample absorbs split-0 FOV noise; cross-fitting removes it (N036)
    fs = pd.read_csv(RES + 'N036-label-position-detrend-v7-cross-fit-the/fov_share.csv')
    pv = fs.pivot_table(index=['target', 'batch'], columns=['version', 'split'], values='fov_share')
    gap = {v: (pv[(v, 0)] - pv[[(v, 1), (v, 2)]].mean(axis=1)) for v in ('v3c', 'v5c', 'v5cx')}
    names = ['None', 'In-sample', 'Cross-fit']
    pfig.raincloud(ax[3], [gap[v].values for v in ('v3c', 'v5c', 'v5cx')], [GREY, pfig.PALETTE[4], SKY], names=names)
    ax[3].set_ylabel('FOV share, split 0 minus held-out'); ax[3].axhline(0, color='black', lw=0.5, zorder=0)
    pfig.stats_vs_ref(ax[3], 'Cross-fit', paired=True)
    # e per-cell background removes the FOV-level label variance (v8 candidate, failed its other QC criteria)
    qc = pd.read_csv(RES + 'N039-label-v8-re-derive-single-antibody-marke/codex/qc_v8.csv')
    pfig.paired(ax[4], qc.frozen_v5_fov_excess.values, qc.fov_excess.values, [GREY, pfig.PALETTE[1]], ['Smoothed bg', 'Per-cell bg'])
    ax[4].set_ylabel('FOV-level label variance share')
    # f the draw-shift cost sits in one direction (N039, data-matched in-distribution minus held-out, per marker)
    ce = pd.read_csv(RES + 'N039-label-v8-re-derive-single-antibody-marke/n039_ceiling_per_target.csv')
    cost = {b: (ce[(ce.arm == 'fov_v5_128k') & (ce.draw == b)].set_index('target').r - ce[(ce.arm == 'goal_base') & (ce.draw == b)].set_index('target').r).sort_index()
            for b in (0, 1)}
    pfig.bar(ax[5], [cost[0].values, cost[1].values], [pfig.PALETTE[2], pfig.PALETTE[5]], names=['Test draw 1', 'Test draw 2'])
    ax[5].axhline(0, color='black', lw=0.5, zorder=0); ax[5].set_ylabel('Seen minus held-out r')
    pfig.stats_pairs(ax[5], [('Test draw 1', 'Test draw 2')], paired=True)
    from scipy.stats import spearmanr; rho = spearmanr(d2.label_fov_excess, d2.r_within - d2.r_pooled)
    return pfig.save(fig, name='fig3_label_limits', sub='paper', root=ROOT,
                     source=[RES + 'N038-review-lf-mk-common-11-batch-held-out-go/d2_within_fov.csv', RES + 'N036-label-position-detrend-v7-cross-fit-the/fov_share.csv',
                             RES + 'N039-label-v8-re-derive-single-antibody-marke/codex/qc_v8.csv', RES + 'N039-label-v8-re-derive-single-antibody-marke/n039_ceiling_per_target.csv'],
                     notes='a-c: final GOAL ensemble (N031), 22 = 11 markers x 2 directions; FOV-level share = variance share of FOV means minus a '
                           '20-permutation null (FOVs >= 3 cells); c Spearman rho %.3f, p %.2g, n 22. d: per target x draw, split-0 minus mean of '
                           'splits 1/2 FOV-mean variance share, label versions v3c (no position correction) / v5c (HGB fitted in-sample on split 0) '
                           '/ v5cx (FOV-fold cross-fit). e: 8 single markers x 2 draws, authors\' smoothed background_lowess (v5) vs per-cell '
                           'background (v8 candidate). f: r50 FOV-split model trained on 128k cells of both draws minus held-out N031 base, '
                           'per marker; test draw 1 = batch 0, test draw 2 = batch 1.' % (rho.statistic, rho.pvalue))


def fig4():
    """Giemsa (Wright) student: same-cell autofluorescence, transitive marker estimate, virtual stains, clinical smears."""
    fig, ax = pfig.figure(2, 3, height_mm=150, height_ratios=[1, 1.35]); ax = ax.ravel()
    met = list(csv.DictReader(open(RES + 'N029-clean-calibration-cohort-no-giemsa-hist/n029_metrics.csv'))) + \
          list(csv.DictReader(open(RES + 'N032-teacher-run-variance-of-clinical-transfe/n032_n033_metrics.csv')))
    two = lambda r: r['arm'] in ('tt6c', 't0', 't1', 't2'); one = lambda r: r['arm'] == 'v4c'
    for k, sl in enumerate(('gs_autofl_within_b0', 'gs_autofl_within_b1')):
        g = [[x for _, x in sorted({(r['run'], float(r['value'])) for r in met if r['metric'] == sl and f(r)})] for f in (one, two)]
        pfig.bar(ax[k], g, [GREY, SKY], names=['One teacher', 'Two teachers'])
        ax[k].set_ylabel(f'Giemsa vs measured autofluor. r, slide {k + 1}'); pfig.stats_vs_ref(ax[k], 'Two teachers')
    # c transitive Giemsa -> marker estimate (two-teacher chain), rings = two directions
    est = collections.defaultdict(list)
    for r in csv.DictReader(open(RES + 'N025-giemsa-chain-without-the-trade-off-two-t/gs_mk_estimates_v4_v5_tt6.csv')):
        if r['chain'] == 'tt6' and r['marker'].startswith(('single', 'grp')) and 'CD123/HLA' not in r['marker']: est[r['marker']].append(float(r['r_est']))
    eo = sorted(est, key=lambda k: np.mean(est[k]))
    pfig.bar(ax[2], [est[k] for k in eo], [pfig.PALETTE[5]] * len(eo), names=[short(k) for k in eo])
    ax[2].set_ylabel('Estimated Giemsa -> marker r'); ax[2].tick_params(axis='x', labelrotation=60); [t.set_ha('right') for t in ax[2].get_xticklabels()]
    pfig.no_stats(ax[2], 'descriptive per-marker estimate r(GS,M) ~ r(S,T) r(T,M); markers not compared')
    # d,e clinical contrasts: AUROC mean and range over 4 LF-teacher runs (student seeds averaged inside a run)
    tr = list(csv.DictReader(open(RES + 'N032-teacher-run-variance-of-clinical-transfe/transfer_all.csv')))
    nice = lambda k: (k.replace('(held-out)', '').replace('neutrophil', 'neut').replace('lymphocyte', 'lym').replace('monocyte', 'mono')
                       .replace('grpCD3/19/56', 'CD3/19/56').replace('erythroblast', 'erythrobl.').replace('basophil', 'baso')
                       .replace('eosinophil', 'eos').replace('myelocyte', 'myelo').replace('normoblast', 'normobl.').strip())
    for k, dset in ((3, 'MLL23'), (4, 'PBC-B')):
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in tr:
            if r['set'] == dset and r['arm'] in ('t0', 't1', 't2', 'old_v4teacher'): per[r['contrast']][r['arm']].append(float(r['auroc']))
        tm = {c: np.array([np.mean(v) for v in d.values()]) for c, d in per.items()}; order = sorted(tm, key=lambda c: -tm[c].mean())
        pfig.forest(ax[k], [nice(c) for c in order], [tm[c].mean() for c in order], [tm[c].min() for c in order], [tm[c].max() for c in order],
                    [pfig.PALETTE[2]] * len(order), ref=0.5)
        ax[k].set_xlabel(f'AUROC, {"MLL23" if dset == "MLL23" else "PBC held-out half"}'); ax[k].set_xlim(0.15, 1.02)
    # f Giemsa student on virtual stains vs real stains: per-marker agreement
    ag = json.load(open(W + '/runs/n028_vs_batch0/downstream/downstream.json'))['agreement']
    ks = sorted([k for k in ag if k != 'autofl' and 'CD123/HLA' not in k], key=lambda k: ag[k]['gen_vs_reg'])
    y = np.arange(len(ks)); ax[5].barh(y, [ag[k]['gen_vs_reg'] for k in ks], height=0.62, color=pfig.PALETTE[4], edgecolor=pfig.stroke(pfig.PALETTE[4]), linewidth=0.5)
    ax[5].set_yticks(y); ax[5].set_yticklabels([short(k) for k in ks]); ax[5].set_xlim(0, 1); ax[5].set_xlabel('Agreement r, virtual vs real Giemsa')
    ax[5].tick_params(axis='y', length=0)
    return pfig.save(fig, name='fig4_giemsa_clinical', sub='paper', root=ROOT,
                     source=[RES + 'N029-clean-calibration-cohort-no-giemsa-hist/n029_metrics.csv', RES + 'N032-teacher-run-variance-of-clinical-transfe/n032_n033_metrics.csv',
                             RES + 'N025-giemsa-chain-without-the-trade-off-two-t/gs_mk_estimates_v4_v5_tt6.csv',
                             RES + 'N032-teacher-run-variance-of-clinical-transfe/transfer_all.csv', W + '/runs/n028_vs_batch0/downstream/downstream.json'],
                     notes='a,b: within-slide r of Giemsa-student autofluorescence prediction vs measured autofluorescence on the same cells, one '
                           'teacher (v4c) vs two teachers (TT6c and its re-trained teacher runs t0-t2), rings = runs, Welch t. c: transitive '
                           'estimate r(S,T) x r(T,M), rings = two directions. d,e: per clinical contrast, square = mean over 4 LF-teacher runs '
                           '(student seeds averaged within a run), line = min-max over the 4 runs (NOT a confidence interval), vertical line = chance .5. '
                           'f: Giemsa-student prediction on virtual Wright images vs on real Wright images of the same cells (N028, one value per marker).')


def fig5():
    """Label-free -> fluorescence marker IMAGES by conditional flow matching (N034/N035), frozen untouched test (N035 primary)."""
    fig, ax0 = pfig.figure(1, 1, height_mm=270); ax0.remove()
    gs = fig.add_gridspec(2, 1, height_ratios=[3.15, 1])
    top = gs[0].subgridspec(8, 7, wspace=0.04, hspace=0.04); bot = gs[1].subgridspec(1, 3)
    MT = np.load(W + '/runs/n035/eval_split1/montage.npy', allow_pickle=True).item()
    cols = ['Brightfield', 'Real', 'Sample 1', 'Sample 2', 'Sample 3', 'Flow mean', 'Regression']
    mag = pfig.cmap('magma')
    for i, (st, lab) in enumerate([('CD16', 'CD16'), ('CD14', 'CD14'), ('CD3', 'CD3'), ('unstained', 'Autofluor.')]):
        m = MT[f'newdraw|{st}']; real = m['real'].astype(np.float32)
        ch = int(np.argmax(real[:, :, 32:96, 32:96].mean((2, 3)).std(0))); tot = real[:, ch].mean((1, 2))
        for k, c in enumerate([int(np.argmax(tot)), int(np.argmin(tot))]):
            imgs = [m['lf'][c, 0], real[c, ch]] + [m['samp'][c, q, ch] for q in range(3)] + [m['mean'][c, ch], m['reg'][c, ch]]
            lo, hi = np.percentile(real[:, ch], [1, 99.7]); r = 2 * i + k
            for j, im in enumerate(imgs):
                a = fig.add_subplot(top[r, j]); a.imshow(np.asarray(im, np.float32), cmap='gray' if j == 0 else mag,
                                                         vmin=None if j == 0 else lo, vmax=None if j == 0 else hi, interpolation='nearest')
                a.set_xticks([]); a.set_yticks([]); [s_.set_visible(False) for s_ in a.spines.values()]
                if (r, j) != (0, 0): a.set_gid(pfig.NO_LETTER)
                if j == 0: a.set_ylabel(f'{lab} {"high" if k == 0 else "low"}', rotation=0, ha='right', va='center')
                if r == 7: a.set_xlabel(cols[j])
    P = pd.read_csv(RES + 'N035-frozen-confirmation-of-n034-a-in-the-unt/n035_per_label.csv')
    P = P[(P['eval'] == 'eval_split1') & (P.cohort == 'newdraw')].sort_values('label')
    FLOW, REG, SHUF, REAL = SKY, GREY, pfig.PALETTE[4], pfig.PALETTE[2]
    ax = [fig.add_subplot(bot[0, j]) for j in range(3)]
    pfig.bar(ax[0], [P.r_avgread.values, P.r_reg.values, P.r_shuffled.values, P.r_realref.values], [FLOW, REG, SHUF, REAL],
             names=['Flow', 'Regression', 'Shuffled', 'Real image'])
    ax[0].set_ylabel('r with marker label'); ax[0].axhline(0, color='black', lw=0.5, zorder=0); pfig.stats_vs_ref(ax[0], 'Flow', paired=True)
    pfig.bar(ax[1], [P.crps_sample.values, P.crps_reg.values], [FLOW, REG], names=['Flow', 'Regression'])
    ax[1].set_ylabel('CRPS (sd units)'); pfig.stats_vs_ref(ax[1], 'Flow', paired=True)
    pfig.bar(ax[2], [P.condw1_sample.values, P.condw1_reg.values, P.condw1_shuffled.values], [FLOW, REG, SHUF], names=['Flow', 'Regression', 'Shuffled'])
    ax[2].set_ylabel('Conditional W1 (sd units)'); pfig.stats_vs_ref(ax[2], 'Flow', paired=True)
    return pfig.save(fig, name='fig5_marker_images_flow', sub='paper', root=ROOT,
                     source=[W + '/runs/n035/eval_split1/eval.json', W + '/runs/n035/eval_split1/montage.npy', RES + 'N035-frozen-confirmation-of-n034-a-in-the-unt/n035_per_label.csv'],
                     notes='Frozen pre-registered test N035: models trained on blood draw 1 (batch 0), tested on untouched draw 2 split 1, Heun 48 '
                           'steps, CFG 2, K 8, 400 cells per stain. a: brightest and dimmest of the first six evaluated cells per stain in that stain\'s '
                           'most variable channel (illustration, not marker calls); one colour scale per row. b-d: 12 labels with a valid readout, '
                           'paired by label; r = readout of generated images vs curated label (flow = mean over K samples), real image = readout of '
                           'the real fluorescence images; CRPS and conditional W1 vs the real images\' readout, lower is better. HLA-DR real-image '
                           'readout does not transfer between draws (r -.11), see N035 post-hoc.')


def fig6():
    """Set-level distribution matching with cell pairing hidden (N034-B v2): readout learnt from bags of cells."""
    fig, ax = pfig.figure(1, 3, height_mm=62); ax = ax.ravel()
    d = pd.read_csv(RES + 'N034-lf-fluorescence-marker-images-by-conditi/setmatch/setmatch_sim_v2.csv'); d = d[d.test == 'newdraw']
    designs = [('same_fov', 'Same cells, FOV sets'), ('same_rand', 'Same cells, random sets'), ('other_fov', 'Other cells, same FOVs'), ('other', 'Other cells, random')]
    cols = [SKY, pfig.PALETTE[1], pfig.PALETTE[3], pfig.PALETTE[4]]
    paired = d[(d['mode'] == 'paired')].groupby('target').r.mean()
    for (mo, lab), c in zip(designs, cols):
        g = d[(d['mode'] == mo) & (d.learner == 'bagmean')].groupby(['n', 'target']).r.mean().unstack()
        pfig.line(ax[0], g.index.values, g.mean(axis=1).values, c, label=lab)
    pfig.threshold(ax[0], y=paired.mean()); pfig.log_axis(ax[0], 'x')
    ax[0].set_xlabel('Cells per set'); ax[0].set_ylabel('r with marker label, new draw'); ax[0].legend(loc='upper right'); ax[0].set_ylim(-0.12, 0.95)
    at = lambda mo, n, ln='bagmean': d[(d['mode'] == mo) & (d.learner == ln) & (d.n == n)].groupby('target').r.mean().reindex(paired.index).values
    pfig.bar(ax[1], [paired.values] + [at(mo, 16) for mo, _ in designs], [GREY] + cols, names=['Paired', 'FOV sets', 'Random sets', 'Other, FOV', 'Other'])
    ax[1].set_ylabel('r at 16 cells per set'); ax[1].axhline(0, color='black', lw=0.5, zorder=0)
    ax[1].tick_params(axis='x', labelrotation=30); [t.set_ha('right') for t in ax[1].get_xticklabels()]; pfig.stats_vs_ref(ax[1], 'Paired', paired=True)
    pfig.bar(ax[2], [at('same_rand', 64, ln) for ln in ('bagmean', 'ot', 'ot_init')], [pfig.PALETTE[1], pfig.PALETTE[5], pfig.PALETTE[2]],
             names=['Bag mean', '1-D OT', 'OT from bag mean'])
    ax[2].set_ylabel('r at 64 cells per set'); ax[2].axhline(0, color='black', lw=0.5, zorder=0)
    ax[2].tick_params(axis='x', labelrotation=30); [t.set_ha('right') for t in ax[2].get_xticklabels()]; pfig.stats_vs_ref(ax[2], 'Bag mean', paired=True)
    return pfig.save(fig, name='fig6_setlevel_matching', sub='paper', root=ROOT,
                     source=RES + 'N034-lf-fluorescence-marker-images-by-conditi/setmatch/setmatch_sim_v2.csv',
                     notes='Linear readout of frozen SSL label-free features (n007, 64 PCs) learnt from set-level information only; 7 targets '
                           '(CD16, CD14, CD45, CD3, HLA-DR, CD19 single stains + autofluorescence), trained on draw 2 split 0, tested on draw 1 '
                           'split 2. a: mean over targets (spread per target in b), dashed line = paired (cell-level) ridge; b,c: rings = '
                           'targets, paired by target.')


def fig1():
    """Architecture: what is trained on what, and what each model predicts. Parameter counts read from the checkpoints
    (n031_base_b10 enc 23.65 M; n029_r18_v6c_fov enc 11.24 M; n029_tt6c_s0 gs_bind vit 85.7 M, freeze 8 of 12; n028 UNet 1.96 M; N034 CondUNet 27.7 M)."""
    from pfig.diagram import Diagram
    d = Diagram(height_mm=114)
    XI, WI, XM, WM, XO, WO = 4, 52, 72, 64, 152, 58          # column x / width (mm) of inputs, models, outputs
    g_in = d.group('Inputs', XI, 8, WI, 102, id='gin')
    lf = d.box('Label-free', XI + 3, 15, WI - 6, 26, sub=['23 LEDs + phase', '[B, 24, 128, 128]', '412,941 cells'], parent=g_in, id='lf')
    fl = d.box('Fluorescence', XI + 3, 44, WI - 6, 16, sub=['6 channels', 'training targets only'], parent=g_in, id='fl')
    gs = d.box('Giemsa (Wright)', XI + 3, 64, WI - 6, 18, sub=['15,712 cells', 'same cells as label-free'], parent=g_in, id='gs')
    cl = d.box('Clinical smears', XI + 3, 88, WI - 6, 18, sub=['PBC, MLL23', 'class anchors'], parent=g_in, id='cl')
    g_mk = d.group('Marker predictor', XM, 8, WM, 30, id='gmk')
    d.box('ResNet-50, full view', XM + 3, 15, WM - 6, 9, sub='23.6 M', parent=g_mk, id='r50a')
    d.box('ResNet-50, 2x zoom', XM + 3, 26, WM - 6, 9, sub='23.6 M', parent=g_mk, id='r50b')
    fm = d.box('Flow matching', XM, 42, WM, 13, sub=['CondUNet, 27.7 M'], fill='solid', id='fm')
    vs = d.box('Virtual stainer', XM, 58, WM, 11, sub=['UNet, 2.0 M'], fill='solid', id='vs')
    g_gc = d.group('Giemsa chain', XM, 72, WM, 38, id='ggc')
    t18 = d.box('Two ResNet-18 teachers', XM + 3, 79, WM - 6, 11, sub='markers + autofluor., 11.2 M each', parent=g_gc, id='t18')
    st = d.box('Giemsa student', XM + 3, 95, WM - 6, 12, sub=['DinoBloom-B, 85.7 M', '4 of 12 blocks trained'], parent=g_gc, id='stu')
    g_out = d.group('Outputs', XO, 8, WO, 102, id='gout')
    o1 = d.box('11 markers', XO + 3, 15, WO - 6, 16, sub=['+ autofluorescence', 'new blood draw'], parent=g_out, id='o1')
    o2 = d.box('Marker images', XO + 3, 42, WO - 6, 13, sub='6 channels', parent=g_out, id='o2')
    o3 = d.box('Virtual Giemsa', XO + 3, 58, WO - 6, 11, parent=g_out, id='o3')
    o4 = d.box('Markers from Giemsa', XO + 3, 90, WO - 6, 17, sub=['+ autofluorescence', 'clinical cell profiles'], parent=g_out, id='o4')
    y_lf, x_lf, x_gs = 39.0, XI + WI + 6, XI + WI + 10           # one label-free trunk feeds the three lower models
    d.arrow(lf, g_mk, ta=(23 - 15) / 26, tb=0.5)                   # horizontal at y 23
    for tgt, yt in ((fm, 45.0), (vs, 63.5), (t18, 84.5)):
        d.arrow(lf, tgt, ta=(y_lf - 15) / 26, via=[(x_lf, y_lf), (x_lf, yt)], tb=(yt - tgt.y) / tgt.h)
    d.arrow(fl, fm, ta=(51 - 44) / 16, tb=(51 - 42) / 13)          # horizontal at y 51
    d.arrow(gs, st, ta=(73 - 64) / 18, via=[(x_gs, 73), (x_gs, 98)], tb=(98 - 95) / 12)
    d.arrow(cl, st, ta=(104 - 88) / 18, tb=(104 - 95) / 12, label='anchors', label_w=13)
    d.arrow(t18, st, sa='b', sb='t', route='v')
    d.arrow(g_mk, o1, route='h'); d.arrow(fm, o2, route='h'); d.arrow(vs, o3, route='h')
    d.arrow(st, o4, ta=(101 - 95) / 12, tb=(101 - 90) / 17)
    return d.save(name='fig1_architecture', sub='paper', root=ROOT,
                  notes='Parameter counts from checkpoints: marker predictor encoders 23.65 M each (+13-output head), Giemsa-chain teachers '
                        'ResNet-18 11.24 M each (markers: v4c labels; autofluorescence: unstained slides, v6c), Giemsa student DinoBloom-B ViT '
                        '85.7 M with 8 of 12 blocks frozen, distilled from the two teachers on the 15,712 Giemsa-label-free pairs and class-'
                        'anchored to PBC half A; virtual stainer UNet 1.96 M trained on 12k registered pairs; flow-matching CondUNet 27.7 M '
                        '(label-free + stain token -> 6-channel fluorescence images).')


FIGS = {'fig1': fig1, 'fig2': fig2, 'fig3': fig3, 'fig4': fig4, 'fig5': fig5, 'fig6': fig6}
if __name__ == '__main__':
    for k in (sys.argv[1:] or list(FIGS)):
        print(k, FIGS[k]())
