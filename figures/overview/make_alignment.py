"""Alignment experiments for an ML audience (BSCCM, 2026-11-24). Run with ~/miniforge3/envs/triad/bin/python.
fig8_alignment_crossmodal: aligning label-free microscope images (LF) with Giemsa-stained photos (GS) of the same cells.
fig9_alignment_domain: domain shift (GS student -> clinical smear datasets PBC / MLL23; LF model -> a new blood draw).
Sources: data/alignment_metrics.csv (extracted from the notebook, every row cites file:line) and the item result files below."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
R = '/home/hpc/v181bb/v181bb10/BSCCM'; RES = R + '/.agent-research/results'; D = R + '/figures/overview/data'
res = lambda n, f: glob.glob(f'{RES}/{n}-*/{f}')[0]
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE; GREY = pfig.BASELINE
A = pd.read_csv(f'{D}/alignment_metrics.csv')

# ---------------------------------------------------------------- Figure 8: LF <-> GS cross-modal alignment
v2 = pd.read_csv(res('N045', 'v2_summary.csv')); v2 = v2[(v2.variant == 'original') & (v2.split == 'test')]
PROT = ['cross_b0to1', 'cross_b1to0']   # held-out GS slide b1 (5,697 cells) / b0 (10,015 cells)
def n045(arm, metric, truth=None):
    s = v2[(v2.arm == arm) & (v2.metric == metric)]
    if truth: s = s[s.truth == truth]
    return np.array([s[s.protocol == p]['mean'].iloc[0] for p in PROT])
ARMS = [('raw_ridge', 'linear map\n(ridge on pairs)', GREY), ('cca', 'CCA', ROSE), ('procrustes', 'Procrustes', ROSE),
        ('siglip_pairs', 'SigLIP\n(contrastive)', ROSE), ('sotalign', 'SOTAlign\n(contrastive + OT)', ROSE)]
TRUTH = 'targets_v5c_position_detrended_autofl'
def n054(method):   # same held-out cells, order b1 then b0
    s = A[(A.item == 'N054') & (A.method == method) & (A.metric == 'same-cell autofluorescence r (cross-slide)')]
    return np.array([s[s.setting.str.startswith('held-out GS slide b1')].value.iloc[0], s[s.setting.str.startswith('held-out GS slide b0')].value.iloc[0]])
fig, ax = pfig.figure(1, 3, height_mm=76, width_ratios=[1, 1.45, 1.1])
pfig.bar(ax[0], [n045(a, 'R10') for a, _, _ in ARMS], [c for _, _, c in ARMS], names=[n for _, n, _ in ARMS])
ax[0].set_ylabel('Exact-cell retrieval, held-out slide (R@10)'); pfig.threshold(ax[0], y=float(n045('cca', 'chance_R10').max()))
pfig.no_stats(ax[0], 'two held-out slides (rings), one seed each')
g = [n045(a, 'GS_head_transfer_pearson_r', TRUTH) for a, _, _ in ARMS]
g += [n054('output distillation (TT6c student, re-evaluated)'), n054('LF hub, latent distillation (GS encoder regresses LF latent)'),
      n054('two-space + unpaired entropic Gromov-Wasserstein flow')]
nm = [n for _, n, _ in ARMS] + ['distill outputs', 'distill latent', 'GW alignment\n(unpaired step)']
pfig.bar(ax[1], g, [c for _, _, c in ARMS] + [SKY, SKY, APR], names=nm)
ax[1].set_ylabel('Transferred per-cell signal (r)'); pfig.no_stats(ax[1], 'two held-out slides (rings), one seed each')
sm = A[(A.item == 'N034') & (A.metric == 'new-draw r of linear readout (mean of 7 targets)')]
ref = sm[sm.method == 'paired readout (pairs visible, reference)'].value.iloc[0]
CUR = [('set-level matching, bag-mean learner, SAME cells, random sets', 'sets of the same cells', SKY),
       ('set-level matching, 1-D OT (sort-matching), SAME cells, random sets', 'same cells, 1-D OT', SEA),
       ('set-level matching, bag-mean, DIFFERENT cells, random sets', 'sets of different cells', ROSE)]
for meth, lab, col in CUR:
    s = sm[sm.method == meth].copy(); s['n'] = s.setting.str.extract(r'n=(\d+)').astype(float)[0]; s = s.sort_values('n')
    pfig.line(ax[2], s.n.values, s.value.values, col, label=lab)
pfig.threshold(ax[2], y=ref); pfig.log_axis(ax[2], 'x')
ax[2].set_xlabel('Set size (cells per supervised set)'); ax[2].set_ylabel('Per-cell r, new blood draw'); ax[2].legend(frameon=False, fontsize=6, loc='upper right', bbox_to_anchor=(1.0, 0.93))
for a in ax[:2]:
    for t in a.get_xticklabels(): t.set_rotation(35); t.set_ha('right'); t.set_rotation_mode('anchor'); t.set_fontsize(6)
pfig.save(fig, name='fig8_alignment_crossmodal', sub='overview', root=R, source=[res('N045', 'v2_summary.csv'), f'{D}/alignment_metrics.csv'],
          notes='Two image modalities of the same blood cells: label-free multi-illumination microscope stacks (LF) and photos of the same '
                'cells after Giemsa staining (GS, the clinical standard); 15.7k registered cell pairs on 2 slides. Goal: make LF-trained '
                'predictors usable on GS images. a exact-cell retrieval GS -> LF on a held-out slide (rings: slide b1, 5,697 cells / slide b0, '
                '10,015 cells; dashed = chance .00176 (b1); b0 chance .00100), N045 v2 (4,500 steps, seed 0). linear map = ridge regression '
                'GS -> LF features fitted on the pairs; SOTAlign = SigLIP pairs + KL-OT on unpaired clinical GS and LF cells. Cross-slide '
                'R@10 .01-.04, while the same SigLIP/SOTAlign models reach .40/.38 and .38/.37 within a slide (held-out fields of view; '
                'chance .010-.016): the failure is the slide-to-slide shift, not missing shared information. b per-cell signal carried '
                'through each complete recipe and compared with its measured value on the same held-out cells (autofluorescence; N045 v5c '
                'and N054 truths correlate 1.000): N045 arms = embedding + ridge readout fitted on the source draw; N054 arms (seed 0) = GS '
                'students trained to regress the LF model\'s outputs (output distillation, TT6c) or its latent (LF hub), and a two-space '
                'model whose final alignment step is unpaired entropic Gromov-Wasserstein (its GS encoder was trained with pairs); N054 LF '
                'training also used unstained cells of the held-out slide. Recipes differ in more than the alignment, so b compares '
                'complete pipelines. c learning from set-level supervision with the pairs hidden (N034-B, user idea, one draw, LF features '
                'from an encoder that saw the test draw): the LF -> label map is learned only from matched SETS of cells (bag means / 1-D '
                'OT); per-cell r on a new blood draw vs set size (mean of 7 targets; 6 at n = 2048, CD19 missing; dashed = fully paired '
                'readout .561). Sets of the same cells recover most of the signal at 4-16 cells and decay with set size; sets of different '
                'cells carry no per-cell information (r ~ 0).')

# ---------------------------------------------------------------- Figure 9: domain shift (clinical smears; new blood draw)
pbc = pd.read_csv(res('N020', 'pbc_transfer_auroc.csv'))   # full-precision per-contrast AUROC, 8 contrasts per arm, same order
P = [('baseline_noDA', 'no adaptation', GREY), ('baseline_noDA_bgwhite', 'test-time\nwhitening', APR),
     ('dann_0.1', 'DANN, lambda .1', ROSE), ('dann_0.3', 'DANN, lambda .3', ROSE)]
CON = list(pbc[pbc.arm == 'baseline_noDA'].contrast)
pv = [pbc[pbc.arm == m].set_index('contrast').value.loc[CON].values for m, _, _ in P]
r58 = pd.read_csv(res('N058', 'n058_per_run.csv')); r56 = pd.read_csv(res('N056', 'n056_per_run.csv')); m57 = pd.read_csv(res('N057', 'plot_n057_mll23.csv'))
TCH = ['t0', 't1', 't2']; m57['teacher'] = m57.seed.str.split('_').str[0]
tm = lambda df, arm, col='MLL23_15', key='arm': df[df[key] == arm].groupby('teacher')[col].mean()[TCH].values
c56 = pd.read_csv(res('N056', 'tables/contrasts_all.csv')); c56 = c56[c56.domain == 'MLL23']
t56 = c56.groupby(['arm', 'teacher', 'seed']).auroc.mean().groupby(['arm', 'teacher']).mean().unstack()[TCH]   # 15 contrasts
post = lambda arm: t56.loc[arm].values
Q = [(tm(r58, 'T'), 'distill outputs', GREY),
     (post('T+OT_tau0.91'), '+ unbalanced OT', APR),
     (post('T+mean'), '+ mean shift', APR),
     (post('T+CORAL'), '+ CORAL', APR),
     (m57[m57.arm == 'anchor 2'].groupby('teacher').value.mean()[TCH].values, 'anchor x 4', SEA),
     (m57[m57.arm == 'anchor 8'].groupby('teacher').value.mean()[TCH].values, 'anchor x 16', SEA),
     (tm(r58, 'H_matched'), 'distill latent', SKY), (tm(r56, 'H'), 'distill latent,\nstronger anchor', SKY)]
assert np.allclose(Q[0][0], post('T'), atol=1e-3) and abs(Q[0][0].mean() - .568) < .001 and abs(Q[-1][0].mean() - .704) < .001 and abs(Q[-2][0].mean() - .681) < .001
ada = A[(A.item == 'N026') & A.metric.str.startswith('GOAL common-11')].copy()
ada['run'] = ada.setting
base = ada[ada.method.str.startswith('no test-time')].set_index('run').value; adab = ada[ada.method.str.startswith('AdaBN')].set_index('run').value
d_ada = (adab - base.loc[adab.index]).values
g14 = pd.read_csv(res('N014', 'n014_goal.csv')).set_index(['arm', 'seed']).value
b13 = pd.read_csv(res('N013', 'n013_v4_noise.csv')).set_index(['protocol', 'seed']).r_mean_12   # full-precision r18 v4 baseline
dann = np.array([g14[('dann', k)] - b13[(k, 0)] for k in ('b1to0', 'b0to1')])
fig, ax = pfig.figure(1, 3, height_mm=72, width_ratios=[1, 1.7, 0.75])
pfig.bar(ax[0], pv, [c for _, _, c in P], names=[n for _, n, _ in P])
ax[0].set_ylabel('PBC smears, 8 contrasts (AUROC)'); pfig.threshold(ax[0], y=0.5)
pfig.stats_vs_ref(ax[0], 'no adaptation', paired=True)
pfig.bar(ax[1], [q[0] for q in Q], [q[2] for q in Q], names=[q[1] for q in Q])
ax[1].set_ylabel('MLL23 smears, 15 contrasts (AUROC)'); ax[1].set_ylim(0.4, 0.82); pfig.threshold(ax[1], y=0.5)
pfig.stats_vs_ref(ax[1], 'distill outputs', paired=True)
pfig.bar(ax[2], [dann, d_ada], [ROSE, APR], names=['DANN', 'AdaBN'])
ax[2].set_ylabel('Change in new-draw r'); pfig.threshold(ax[2], y=0)
pfig.no_stats(ax[2], 'differences vs the seed-matched unadapted model per direction / run; 2 (DANN) and 4 (AdaBN) values')
for a in ax:
    for t in a.get_xticklabels(): t.set_rotation(35); t.set_ha('right'); t.set_rotation_mode('anchor'); t.set_fontsize(6)
pfig.save(fig, name='fig9_alignment_domain', sub='overview', root=R,
          source=[f'{D}/alignment_metrics.csv', res('N056', 'n056_per_run.csv'), res('N057', 'plot_n057_mll23.csv'), res('N058', 'n058_per_run.csv')],
          notes='Domain shift. a-b a network that reads Giemsa photos (GS student, trained on BSCCM Giemsa cells) applied to public clinical '
                'smear datasets (other labs, stains, cameras); quality = AUROC of pre-registered cell-class contrasts (0.5 = chance). a PBC '
                '(N020, seed 0, pbc_transfer_auroc.csv; rings = the 8 contrasts, paired t vs no adaptation, Holm): means .406 / .438 / .462 / '
                '.555; unsupervised domain-adversarial training (DANN, unlabelled PBC half A) and test-time whitening do not transfer '
                'reliably (contrasts .11-.90). b MLL23, 15 contrasts (N056-N058 harness; rings = 3 LF teacher models, each the mean of its '
                '2-3 student seeds; paired t by teacher vs output distillation, Holm). All arms use class anchoring (cross-entropy on 3 '
                'common classes of labelled PBC half A; anchor x 4 / x 16 = loss weight .5 -> 2 / 8). Post-hoc alignment of the clinical '
                'features to BSCCM (N056 tables/contrasts_all.csv; unbalanced OT tau .91, mean shift, CORAL) moves the 3-teacher mean by '
                '<= .007 but single teachers by up to -.038 (OT, teacher 0); the aligned arms decode a normalised 8-view mean embedding '
                'while the raw arm averages 8 decoded predictions. Distilling the LF latent instead of the outputs: .681 (matched '
                'anchoring, N058) / .704 (N056 anchoring). Registered rules: latent vs output distillation = higher (N058, mean gain .113 > '
                'seed floor .074); anchor sweep = no difference. c LF model applied to a new blood draw, change vs the seed-matched '
                'unadapted model: DANN batch-adversarial training (N014, ResNet-18, v4 labels, 12-target mean; 2 directions +.0025 / '
                '-.0015 vs the full-precision seed-0 baseline in n013_v4_noise.csv); AdaBN (re-estimated BatchNorm statistics on 20k '
                'unlabelled new-draw images, N026, 4 runs) -.005 to -.011.')
print('done')
