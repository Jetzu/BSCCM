"""Figures 14-15 (N097 part 2, user 11-27): pairwise separability of all lineages; measured vs predicted scatter with Pearson r for the
lineage-defining markers. Run with ~/miniforge3/envs/triad/bin/python after .agent-research/results/N097-*/n097_separation.py."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
R = '/home/hpc/v181bb/v181bb10/BSCCM'; N = glob.glob(R + '/.agent-research/results/N097-*')[0]
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE

# ------------------------------------------------------------------ Figure 14: pairwise lineage separability (AUROC heatmaps)
b = pd.read_csv(N + '/sep_bsccm.csv'); s = pd.read_csv(N + '/sep_sandbox.csv')
BL = ['T', 'B', 'NK', 'monocyte', 'neutrophil', 'basophil/pDC']
BN = {'T': 'T (CD3)', 'B': 'B (CD19)', 'NK': 'NK (CD56)', 'monocyte': 'monocyte (CD14)', 'neutrophil': 'neutrophil (CD16)', 'basophil/pDC': 'basophil/pDC (CD123)'}
SL = ['CD4 T', 'CD8 T', 'B', 'NK', 'monocyte', 'neutrophil', 'eosinophil']
SN = {'CD4 T': 'CD4 T (CD4)', 'CD8 T': 'CD8 T (CD8)', 'B': 'B (CD19)', 'NK': 'NK (CD56)', 'monocyte': 'monocyte (CD14)', 'neutrophil': 'neutrophil (CD15)',
      'eosinophil': 'eosinophil (Siglec8)'}


def mat(df, labs):
    g = df.groupby(['A', 'B']).auroc.mean()
    return np.array([[g.get((a, c), np.nan) if a != c else np.nan for c in labs] for a in labs])


fig, ax = pfig.figure(1, 3, height_mm=82)
pfig.heatmap(ax[0], mat(b, BL), [BN[x] for x in BL], [BN[x] for x in BL], vmin=.5, vmax=1, label='AUROC (.5 = chance, 1 = perfect)')
pfig.heatmap(ax[1], mat(s[s.arm == 'all markers per cell'], SL), [SN[x] for x in SL], [SN[x] for x in SL], vmin=.5, vmax=1, label='AUROC, all-marker training')
pfig.heatmap(ax[2], mat(s[s.arm == 'one marker per cell'], SL), [SN[x] for x in SL], [SN[x] for x in SL], vmin=.5, vmax=1, label='AUROC, one-marker training')
for a in ax:
    for t in a.get_xticklabels() + a.get_yticklabels(): t.set_fontsize(5.5)
sc = b.groupby(['A', 'B']).slide_control.mean()
pfig.save(fig, name='fig14_lineage_separability', sub='overview', root=R, source=[N + '/sep_bsccm.csv', N + '/sep_sandbox.csv', N + '/n097_separation.py'],
          notes='N097. How well can every pair of cell lineages be told apart from the PREDICTED markers? Scale: for lineages A and B with '
                'defining markers a and b, every cell gets the score z(predicted a) - z(predicted b) (z-score over all test cells); the tile '
                'is the AUROC of that score for A cells vs B cells = the probability that a randomly drawn A cell scores higher than a randomly '
                'drawn B cell. .5 = coin flip (not separable), .7 = weak, .8 = moderate, .9 = good, 1 = perfect. No classifier is fitted; the '
                'matrix is symmetric. a BSCCM, N085 record ensemble, mean of the 2 directions; lineages = positives of each marker on its own '
                'single-stain slide of the test draw. Range .82 (T vs B) to .99 (NK vs monocyte). Because the two lineages of a pair come from '
                'different slides, a slide control scores ALL cells of slide a vs ALL cells of slide b with the same score (same blood draw -> '
                f'expected .5): mean |AUROC - .5| = {abs(sc - .5).mean():.3f}, range {sc.min():.2f}-{sc.max():.2f} (lowest for pairs with the CD16 '
                'slide). b, c ImageStream sandbox (external dataset; all 9 markers measured on every cell; lineages = gates on the held-out '
                'donor\'s measured labels; mean of 3 held-out donors; N080 ResNet-50 models trained with all markers per cell (b) or one '
                'marker per cell (c), the BSCCM-like regime). The measured markers give AUROC 1.00 for every pair (by construction). '
                'Myeloid vs lymphoid pairs are separated well (.88-1.00); lymphocyte subsets are not: CD4 T vs CD8 T .65/.60, CD8 T vs NK '
                '.57/.65, B vs CD8 T .67/.73.')

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
print('done')
