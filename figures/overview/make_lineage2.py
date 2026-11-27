"""Figures 14-15 (N097 part 2, user 11-27): pairwise separability of all lineages; measured vs predicted scatter with Pearson r for the
lineage-defining markers. Run with ~/miniforge3/envs/triad/bin/python after .agent-research/results/N097-*/n097_separation.py."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
R = '/home/hpc/v181bb/v181bb10/BSCCM'; N = glob.glob(R + '/.agent-research/results/N097-*')[0]
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE

# ------------------------------------------------------------------ Figure 14: pairwise lineage separability (AUROC heatmaps)
b = pd.read_csv(N + '/sep_bsccm.csv'); s_sb = pd.read_csv(N + '/sep_sandbox.csv')
BL = ['T', 'B', 'NK', 'monocyte', 'neutrophil', 'basophil/pDC']
BN = {'T': 'T (CD3)', 'B': 'B (CD19)', 'NK': 'NK (CD56)', 'monocyte': 'monocyte (CD14)', 'neutrophil': 'neutrophil (CD16)', 'basophil/pDC': 'basophil/pDC (CD123)'}
SL = ['T', 'B', 'NK', 'monocyte', 'neutrophil', 'eosinophil']
SN = {'T': 'T (CD3)', 'B': 'B (CD19)', 'NK': 'NK (CD56)', 'monocyte': 'monocyte (CD14)', 'neutrophil': 'neutrophil (CD15)', 'eosinophil': 'eosinophil (Siglec8)'}


def mat(df, labs):
    g = df.groupby(['A', 'B']).auroc.mean()
    return np.array([[g.get((a, c), np.nan) if a != c else np.nan for c in labs] for a in labs])


fig, ax = pfig.figure(1, 3, height_mm=82)
for k, (dd, lab) in enumerate((('b10', 'train draw 1, test draw 0'), ('b01', 'train draw 0, test draw 1'), (None, 'mean of both directions'))):
    sub = b if dd is None else b[b.direction == dd]
    pfig.heatmap(ax[k], mat(sub, BL), [BN[x] for x in BL], [BN[x] for x in BL], vmin=.5, vmax=1, label=f'AUROC, {lab}')
for a in ax:
    for t in a.get_xticklabels() + a.get_yticklabels(): t.set_fontsize(5.5)
sc = b.groupby(['A', 'B']).slide_control.mean()
pfig.save(fig, name='fig14_lineage_separability', sub='overview', root=R, source=[N + '/sep_bsccm.csv', N + '/n097_separation.py'],
          notes='N097. BSCCM: how well can every pair of cell lineages be told apart from the PREDICTED markers (N085 record ensemble)? '
                'Scale: for lineages A and B with defining markers a and b, every cell gets the score z(predicted a) - z(predicted b) '
                '(z-score over all test cells); the tile is the AUROC of that score for A cells vs B cells = the probability that a randomly '
                'drawn A cell scores higher than a randomly drawn B cell. .5 = coin flip (not separable), .7 = weak, .8 = moderate, .9 = good, '
                '1 = perfect. No classifier is fitted; the matrix is symmetric. Lineages = positives of each marker on its own single-stain '
                'slide of the TEST draw. a train draw 1 -> test draw 0 (T vs B .95), b train draw 0 -> test draw 1 (T vs B .70; only 10 B '
                'cells), c mean of a and b (T vs B .82; range .82-.99). Slide control (ALL cells of slide a vs ALL cells of slide b, same '
                f'score; same blood draw -> expected .5): mean |AUROC - .5| = {abs(sc - .5).mean():.3f}, range {sc.min():.2f}-{sc.max():.2f}.')

# ------------------------------------------------------------------ Figure 17: same separability in the ImageStream sandbox
fig, ax = pfig.figure(1, 2, height_mm=82)
pfig.heatmap(ax[0], mat(s_sb[s_sb.arm == 'all markers per cell'], SL), [SN[x] for x in SL], [SN[x] for x in SL], vmin=.5, vmax=1, label='AUROC, all-marker training')
pfig.heatmap(ax[1], mat(s_sb[s_sb.arm == 'one marker per cell'], SL), [SN[x] for x in SL], [SN[x] for x in SL], vmin=.5, vmax=1, label='AUROC, one-marker training')
for a in ax:
    for t in a.get_xticklabels() + a.get_yticklabels(): t.set_fontsize(5.5)
pfig.save(fig, name='fig17_lineage_separability_sandbox', sub='overview', root=R, source=[N + '/sep_sandbox.csv', N + '/n097_separation.py'],
          notes='N097. Same score and AUROC scale as fig14, in the ImageStream sandbox (external dataset; all 9 markers measured on every '
                'cell; lineages = gates on the held-out donor\'s measured labels, all T cells merged into one T lineage scored by CD3; mean of 3 '
                'held-out donors; N080 ResNet-50 models trained with all markers per cell (a) or one marker per cell (b), the BSCCM-like '
                'regime). The measured markers give AUROC 1.00 for every pair (by construction). Myeloid vs lymphoid pairs .85-1.00; T vs B '
                '.72 (a; donors .75/.94/.46) and .74 (b; .81/.91/.52); B vs NK .72/.79; T vs NK .79/.79.')

# ------------------------------------------------------------------ Figure 15: measured vs predicted, lineage-defining markers (BSCCM)
d = pd.read_csv(N + '/scatter_bsccm.csv')
MK = ['CD3', 'CD19', 'CD56', 'CD14', 'CD16', 'CD123', 'HLA-DR', 'CD45']
DIRS = [('b10', 'train draw 1, test draw 0', SKY), ('b01', 'train draw 0, test draw 1', APR)]
fig, ax = pfig.figure(2, 4, height_mm=120)
rng = np.random.default_rng(0); rvals = {}
for k, m in enumerate(MK):
    a = ax.ravel()[k]
    for dd, lab, c in DIRS:
        g = d[(d.direction == dd) & (d.marker == m)]
        r = np.corrcoef(g.measured, g.predicted)[0, 1]; rvals[(m, dd)] = (r, len(g))
        sh = g.iloc[rng.permutation(len(g))[:800]]
        pfig.scatter(a, sh.measured.values, sh.predicted.values, c, label=f'{dd[1]}→{dd[2]}: r = {r:.2f} (n = {len(g):,})', s=2)
    a.set_xlabel(f'Measured {m} (label)'); a.set_ylabel(f'Predicted {m}')
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + .38 * (hi - lo))   # headroom so the r legend never covers points
    a.legend(frameon=False, fontsize=5, loc='upper left', handletextpad=.2, borderaxespad=.1)
rtxt = '; '.join(f'{m} {rvals[(m, "b10")][0]:.2f}/{rvals[(m, "b01")][0]:.2f}' for m in MK)
pfig.save(fig, name='fig15_marker_correlation', sub='overview', root=R, source=[N + '/scatter_bsccm.csv', N + '/n097_separation.py'],
          notes='N097. Measured vs predicted intensity of the lineage-defining markers on BSCCM, N085 record ensemble. Each panel: the cells '
                'of that marker\'s single-stain slide in the test draw; blue = train draw 1 -> test draw 0 (legend "1->0"), orange = train '
                'draw 0 -> test draw 1. Units: the v10 label scale (asinh-transformed fluorescence, z-scored per slide; the vertical stack at '
                'the left is the clipped background floor) and the model\'s prediction on the same scale. Legend: Pearson r over ALL test '
                f'cells of the slide (n), display subsampled to 800 cells per direction. r (1->0 / 0->1): {rtxt}. These are the per-target r '
                'values that enter the GOAL (CD123/HLA-DR are single markers here; the GOAL also includes 3 pooled targets).')
# ------------------------------------------------------------------ Figure 16: CD3 vs CD19 on the same cells (anticorrelation)
sc = pd.read_csv(N + '/sandbox_lymph_cells.csv'); sr = pd.read_csv(N + '/sandbox_cd3_cd19.csv'); bc = pd.read_csv(N + '/cd3cd19_bsccm_cells.csv')
fig, ax = pfig.figure(2, 3, height_mm=135)
def rr(x, y): return np.corrcoef(x, y)[0, 1]
for k, (arm, lab) in enumerate((('truth', 'measured'), ('all markers per cell', 'predicted, all-marker training'), ('one marker per cell', 'predicted, one-marker training'))):
    g = sc[sc.arm == arm]; a = ax[0, k]; pfig.density(a, g.CD3.values, g.CD19.values, gridsize=40)
    rd = sr[sr.arm == arm].sort_values('donor').r_CD3_CD19_lymph.values
    a.plot([], [], ' ', label=f'donor 0 (shown): r = {rd[0]:.2f}'); a.plot([], [], ' ', label=f'donors 1, 2: r = {rd[1]:.2f}, {rd[2]:.2f}')
    a.set_xlim(-2.5, 3); a.set_ylim(-1.5, 6); a.legend(frameon=False, fontsize=5, loc='upper right', handlelength=0, handletextpad=0)
    a.set_xlabel(f'CD3 intensity, z-score ({lab})', fontsize=6); a.set_ylabel(f'CD19 intensity, z-score ({lab})', fontsize=6)
rng = np.random.default_rng(0)
a = ax[1, 0]
for d, lab, c in (('b10', '1\u21920', SKY), ('b01', '0\u21921', APR)):
    g = bc[bc.direction == d]; sh = g.iloc[rng.permutation(len(g))[:1500]]
    pfig.scatter(a, sh.pred_CD3.values, sh.pred_CD19.values, c, label=f'{lab}, all cells: r = {rr(g.pred_CD3, g.pred_CD19):+.2f}', s=2)
a = ax[1, 1]; g = bc[(bc.direction == 'b10') & bc.lymphocyte]
pfig.scatter(a, g.pred_CD3.values, g.pred_CD19.values, SKY, label=f'1\u21920, gated lymphocytes: r = {rr(g.pred_CD3, g.pred_CD19):+.2f}', s=3)
a = ax[1, 2]; g = bc[(bc.direction == 'b10') & (bc.lineage != 'other')]
for lin, c in (('T', SKY), ('B', ROSE)):
    h = g[g.lineage == lin]; pfig.scatter(a, h.pred_CD3.values, h.pred_CD19.values, c, label=f'true {lin} cells (n = {len(h)})', s=4)
a.plot([], [], ' ', label=f'1\u21920, T + B: r = {rr(g.pred_CD3, g.pred_CD19):+.2f}')
for a in ax[1]:
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + .35 * (hi - lo))
    a.legend(frameon=False, fontsize=5, loc='upper right', handletextpad=.2); a.set_xlabel('Predicted CD3 (label scale)'); a.set_ylabel('Predicted CD19 (label scale)')
pfig.save(fig, name='fig16_cd3_cd19_anticorrelation', sub='overview', root=R,
          source=[N + '/sandbox_lymph_cells.csv', N + '/sandbox_cd3_cd19.csv', N + '/cd3cd19_bsccm_cells.csv', N + '/n097_cd3cd19.py'],
          notes='N097. CD3 vs CD19 on the SAME cells (T vs B; the anticorrelation question). Fig 15 plots each marker against itself (measured '
                'vs predicted CD3), which is an accuracy plot and must be positive; anticorrelation is a property of two DIFFERENT markers on '
                'one cell. a-c ImageStream sandbox, lymphocytes of held-out donor 0 (both markers measured on every cell): measured (a) T and B '
                'cells form two clouds, r = -.63 (donors 1, 2: -.88, -.68); predicted with all-marker (b) or one-marker (c) training: one '
                'anti-diagonal band, r -.84..-.93 (more anticorrelated than measured). d-f BSCCM, record ensemble: no BSCCM cell has both CD3 '
                'and CD19 measured, so only predictions can be plotted. d all test cells (display 1,500 per direction): r +.23 / +.18 - '
                'positive, because both markers are lymphocyte markers and the majority of white cells (granulocytes, monocytes) is low on '
                'both. e gated lymphocytes (draw-0 all-marker slide, 371 cells, train draw 1 -> test draw 0): r -.53. f true T (CD3+ on the CD3 '
                'slide) and true B (CD19+ on the CD19 slide) cells of the test draw 0: r -.49. Label scale = asinh fluorescence z-scored per slide.')
print('done')
