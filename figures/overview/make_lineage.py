"""Figures 11-12 (N097, user 11-27): are predicted CD3 and CD19 exclusive, and can the models separate cell lineages?
Run with ~/miniforge3/envs/triad/bin/python after .agent-research/results/N097-*/n097_lineage.py (bsccm.sif) wrote the CSVs."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
R = '/home/hpc/v181bb/v181bb10/BSCCM'; N = glob.glob(R + '/.agent-research/results/N097-*')[0]
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE; GREY = pfig.BASELINE
DIV = pfig.cmap('house', kind='diverging')

# ------------------------------------------------------------------ Figure 11: BSCCM (record ensemble E8, both directions)
prof = pd.read_csv(N + '/bsccm_lineage_profiles.csv'); auc = pd.read_csv(N + '/bsccm_auroc.csv')
cor = pd.read_csv(N + '/bsccm_cd3_cd19_corr.csv'); cells = pd.read_csv(N + '/bsccm_tb_cells.csv')
ROWS = [('CD3', 'T cells (CD3+)'), ('CD19', 'B cells (CD19+)'), ('CD56', 'NK cells (CD56+)'), ('CD14', 'monocytes (CD14+)'),
        ('CD16', 'CD16+ cells'), ('CD123', 'CD123+ cells')]
COLS = ['CD3', 'CD19', 'CD56', 'CD14', 'CD16', 'CD123', 'HLA-DR', 'CD45']
e8 = prof[(prof.model == 'E8 record')]
H = np.array([[e8[(e8.lineage == x) & (e8.marker == y)].mean_pred_z.mean() for y in COLS] for x, _ in ROWS])
fig, ax = pfig.figure(2, 3, height_mm=150)
pfig.heatmap(ax[0, 0], H, [n for _, n in ROWS], ['pred. ' + c for c in COLS], cmap=DIV, vmin=-2, vmax=2, label='Mean predicted z-score')
for k, d in enumerate(('b10', 'b01')):
    for grp, c in (('T (CD3+ slide)', SKY), ('B (CD19+ slide)', ROSE)):
        s_ = cells[(cells.group == grp) & (cells.direction == d)]
        pfig.scatter(ax[0, 1 + k], s_.pred_CD3.values, s_.pred_CD19.values, c, label=grp.split(' ')[0] + ' cells', s=4)
    ax[0, 1 + k].set_xlabel('Predicted CD3 (z)'); ax[0, 1 + k].set_ylabel('Predicted CD19 (z)'); ax[0, 1 + k].legend(frameon=False, fontsize=6, loc='upper right')
TESTS = [('CD3+ vs CD3- (own slide)', 'CD3+ vs -'), ('CD19+ vs CD19- (own slide)', 'CD19+ vs -'), ('CD56+ vs CD56- (own slide)', 'CD56+ vs -'),
         ('CD14+ vs CD14- (own slide)', 'CD14+ vs -'), ('CD16+ vs CD16- (own slide)', 'CD16+ vs -'), ('CD123+ vs CD123- (own slide)', 'CD123+ vs -'),
         ('B vs T by predicted CD19 - CD3', 'B vs T')]
a8 = auc[auc.model == 'E8 record']
pfig.bar(ax[1, 0], [a8[a8.test == t].sort_values('direction').auroc.values for t, _ in TESTS], [SKY] * 6 + [ROSE], names=[n for _, n in TESTS])
pfig.threshold(ax[1, 0], y=.5); ax[1, 0].set_ylim(.4, 1.02); ax[1, 0].set_ylabel('AUROC (record ensemble)')
pfig.no_stats(ax[1, 0], 'one AUROC per direction (rings = 2 directions); dashed = chance')
c8 = cor[cor.model == 'E8 record']
CG = [('all test cells', 'all cells', GREY), ('gated lymphocytes', 'gated\nlymphocytes', SKY), ('true T + B cells', 'true T + B\ncells', ROSE)]
pfig.bar(ax[1, 1], [c8[c8.cells == g].r.values for g, _, _ in CG], [c for _, _, c in CG], names=[n for _, n, _ in CG])
pfig.threshold(ax[1, 1], y=0); ax[1, 1].set_ylabel('r(predicted CD3, predicted CD19)'); ax[1, 1].set_ylim(-.7, .4)
pfig.no_stats(ax[1, 1], 'one r per direction (rings); gated lymphocytes exist only on the draw-0 all-marker slide (1 value)')
MEM = [('E8 record', 'record ens.'), ('n048_v9_base', 'R50'), ('n048_v9_zoom2', 'R50 zoom'), ('n053_cytoland', 'Cytoland'),
       ('comp_base', 'comp.'), ('comp_z2', 'comp. zoom'), ('comp_cyto', 'comp. Cyto'), ('cnxs_plain', 'CNX-S'), ('cnxs_comp', 'CNX-S comp.')]
bt = auc[auc.test == 'B vs T by predicted CD19 - CD3']
pfig.bar(ax[1, 2], [bt[bt.model == m].sort_values('direction').auroc.values for m, _ in MEM], [SKY] + [GREY] * 8, names=[n for _, n in MEM])
pfig.threshold(ax[1, 2], y=.5); ax[1, 2].set_ylim(.4, 1.02); ax[1, 2].set_ylabel('AUROC B vs T (pred. CD19 - CD3)')
pfig.no_stats(ax[1, 2], 'one AUROC per direction (rings); b01 has only 10 B cells')
for a in (ax[1, 0], ax[1, 1], ax[1, 2]):
    for t in a.get_xticklabels(): t.set_rotation(45 if a is ax[1, 2] else 30); t.set_ha('right'); t.set_rotation_mode('anchor'); t.set_fontsize(6)
nb = cells.groupby(['direction', 'group']).size().to_dict()
pfig.save(fig, name='fig11_lineages_bsccm', sub='overview', root=R,
          source=[N + '/bsccm_lineage_profiles.csv', N + '/bsccm_auroc.csv', N + '/bsccm_cd3_cd19_corr.csv', N + '/bsccm_tb_cells.csv', N + '/n097_lineage.py'],
          notes='N097. BSCCM never measures two single markers on one cell, so lineages are defined per single-stain slide of the TEST draw: '
                'positives = label above the 2-component GMM valley (CD16) or above max(valley, median + 3 MAD) for the rare-positive markers '
                '(evaluation only). Model = N085 record ensemble E8 (GOAL .6967), both directions. a mean predicted z-score (z over all test '
                'cells of a direction, averaged over the 2 directions) of every marker on the true positives of each lineage: T cells get high '
                'predicted CD3 and low CD19/CD14/CD16, monocytes high CD14/HLA-DR, CD16+ cells (neutrophils) high CD16; B cells are not '
                'cleanly separated from T cells. b, c predicted CD3 vs CD19 on true T (CD3+ on the CD3 slide) and true B (CD19+ on the CD19 slide) '
                f'cells, b = train draw 1 -> test draw 0 (b10), c = b01 (counts {nb}); in c 67 of 692 measured-CD3+ cells get predicted CD19 z > 2 '
                'and predicted HLA-DR above the other T cells, spread over 63 of 249 fields of view with clearly positive CD3 labels (median 1.67) -> '
                'a model error (B-like T cells), not a field-of-view label artefact. d AUROC: each marker detects its own positives (CD3 .99/.86, CD14 .98/.97, '
                'CD16 .97/.97; CD19 .83/.87, CD123 .87/.86 for b10/b01); B vs T by predicted CD19 - CD3 .95 (b10, 39 B vs 89 T) and .70 (b01, '
                'only 10 B vs 692 T). e correlation of the predicted CD3 and CD19 outputs: positive over all cells (both are lymphocyte markers, '
                '+.23/+.18), negative within gated lymphocytes (-.54, draw 0 only) and within true T+B cells in b10 (-.49), ~0 in b01 (+.01). '
                'f B vs T AUROC for the record ensemble and each of its 8 members (rings = directions; R50 = ResNet-50, comp. = composite heads, CNX-S = ConvNeXt-S).')

# ------------------------------------------------------------------ Figure 12: ImageStream sandbox (all 9 markers on every cell)
sb = pd.read_csv(N + '/sandbox_cd3_cd19.csv'); sp = pd.read_csv(N + '/sandbox_type_profiles.csv'); sc = pd.read_csv(N + '/sandbox_lymph_cells.csv')
fig, ax = pfig.figure(3, 3, height_mm=200)
for k, (arm, lab) in enumerate((('truth', 'measured'), ('all markers per cell', 'predicted, all-marker training'),
                                ('one marker per cell', 'predicted, one-marker training'))):
    s = sc[sc.arm == arm]; pfig.density(ax[0, k], s.CD3.values, s.CD19.values, gridsize=40)
    ax[0, k].set_xlabel(f'CD3 z ({lab})', fontsize=6); ax[0, k].set_ylabel(f'CD19 z ({lab})', fontsize=6)
    ax[0, k].set_xlim(-2.5, 3); ax[0, k].set_ylim(-1.5, 4)
ARMS = [('truth', 'measured', GREY), ('all markers per cell', 'all-marker\ntraining', SKY), ('one marker per cell', 'one-marker\ntraining', APR)]
ARMS += [(a, a.replace('one marker + corr loss ', 'one-marker +\ncorr. loss ').replace('all markers + corr loss ', 'all-marker +\ncorr. loss '),
          ROSE if a.startswith('one') else SEA) for a in sb.arm.unique() if 'corr loss' in a]
vals = [sb[sb.arm == a].sort_values('donor').r_CD3_CD19_lymph.values for a, _, _ in ARMS]
pfig.bar(ax[1, 0], vals, [c for _, _, c in ARMS], names=[n for _, n, _ in ARMS]); ax[1, 0].set_ylabel('r(CD3, CD19) in lymphocytes')
pfig.stats_vs_ref(ax[1, 0], 'measured', paired=True)
A2 = ARMS[1:]
pfig.bar(ax[1, 1], [sb[sb.arm == a].sort_values('donor').auroc_B_vs_T.values for a, _, _ in A2], [c for _, _, c in A2], names=[n for _, n, _ in A2])
pfig.threshold(ax[1, 1], y=.5); ax[1, 1].set_ylabel('AUROC B vs T (pred. CD19 - CD3)'); ax[1, 1].set_ylim(.3, 1.02)
pfig.stats_vs_ref(ax[1, 1], 'all-marker\ntraining', paired=True)
# per-marker accuracy of CD3 and CD19 over all held-out cells (explains the donor that fails)
PM = [(m, a, f'{m}, {n}', c) for m in ('CD3', 'CD19') for a, n, c in (('all markers per cell', 'all-marker', SKY), ('one marker per cell', 'one-marker', APR))]
pfig.bar(ax[1, 2], [sb[sb.arm == a].sort_values('donor')[f'r_{m}_vs_measured'].values for m, a, _, _ in PM], [c for *_, c in PM], names=[n for _, _, n, _ in PM])
ax[1, 2].set_ylabel('Pearson r, predicted vs measured'); pfig.stats_pairs(ax[1, 2], [('CD3, all-marker', 'CD19, all-marker'), ('CD3, one-marker', 'CD19, one-marker')], paired=True)
for a in (ax[1, 0], ax[1, 1], ax[1, 2]):
    for t in a.get_xticklabels(): t.set_rotation(30); t.set_ha('right'); t.set_rotation_mode('anchor'); t.set_fontsize(6)
TY = ['T4', 'T8', 'B', 'NK', 'mono', 'neut', 'eos']; MK = ['CD3', 'CD4', 'CD8', 'CD19', 'CD56', 'CD14', 'CD15', 'Siglec8', 'CD45']
for k, (arm, lab) in enumerate((('truth', 'measured'), ('all markers per cell', 'all-marker'), ('one marker per cell', 'one-marker'))):
    s = sp[sp.arm == arm]
    Hm = np.array([[s[(s.type == t) & (s.marker == m)].mean_z.mean() for m in MK] for t in TY])
    pfig.heatmap(ax[2, k], Hm, TY, MK, cmap=DIV, vmin=-2, vmax=2, label=f'Mean z ({lab})')
pfig.save(fig, name='fig12_lineages_sandbox', sub='overview', root=R,
          source=[N + '/sandbox_cd3_cd19.csv', N + '/sandbox_type_profiles.csv', N + '/sandbox_lymph_cells.csv', N + '/n097_lineage.py'],
          notes='N097. ImageStream sandbox (S-BIAD452: brightfield/side-scatter images, 9 markers measured on EVERY cell, 3 donors, each held '
                'out once; N080 ResNet-50 models). Types = hierarchical gates on the held-out donor\'s own measured labels (evaluation only). '
                'a-c lymphocytes of held-out donor 0: measured CD3 vs CD19 (T and B cells form two separate clouds) vs predictions of the '
                'all-marker and one-marker models (one continuous anti-diagonal band). d r(CD3, CD19) within lymphocytes, rings = 3 donors: '
                'measured -.63/-.88/-.68; predictions are MORE anticorrelated (-.84..-.93): the models encode a single lymphocyte axis with '
                'CD3 and CD19 at opposite ends. e but where a cell sits on that axis tracks its true lineage only partly: AUROC B vs T .74/.94/.46 '
                '(all-marker) and .81/.90/.51 (one-marker); on donor 2 predicted CD19 has r .09 with the measured CD19, so B vs T is at chance. '
                'f Pearson r of predicted vs measured CD3 and CD19 over all held-out cells (rings = donors; paired t, Holm). g-i mean z of every marker per gated type: measured vs predicted; predictions reproduce the granulocyte/monocyte/lymphocyte '
                'split but compress the lymphocyte subtypes (T4/T8/B/NK) toward each other. Paired t vs measured (d) / all-marker (e), Holm (3 donors: low power). '
                'Corr-loss arms (N096) are added when present.')
print('done')
