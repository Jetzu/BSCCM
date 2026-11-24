"""Architecture figures for an ML audience (BSCCM, 2026-11-24). Run with ~/miniforge3/envs/triad/bin/python.
fig6_architecture: the best measured model (N084: 4-member ensemble; the registered best N053 = first three members).
fig7_architectures: every encoder architecture tested on the same protocol (v10 GOAL) + the ensemble build-up.
Numbers: N051/N053/N078/N083/N084 result CSVs (paths below); dims/params measured from the code (timm, bsccm/encoders.py)."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
from pfig.diagram import Diagram
R = '/home/hpc/v181bb/v181bb10/BSCCM'; RES = R + '/.agent-research/results'
res = lambda n, f: glob.glob(f'{RES}/{n}-*/{f}')[0]
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE; GREY = pfig.BASELINE

# ---------------------------------------------------------------- Figure 6: architecture of the best model
d = Diagram(height_mm=80)
g_in = d.group('Input: one cell', 4, 14, 38, 56, id='in')
d.box('Image stack', 7, 22, 32, 20, sub=['[B, 24, 128, 128]', '23 illumination angles', '+ 1 phase image'], parent=g_in, id='x')
d.box('Augment', 7, 48, 32, 15, sub=['shift, scale', 'no rotation / flip'], parent=g_in, id='aug')
g_m = d.group('4 ensemble members (trained separately)', 50, 6, 74, 70, id='mem')
M = [('ResNet-50, full view', ['self-supervised init', '23.6 M, 2048-d']),
     ('ResNet-50, 2x centre zoom', ['self-supervised init', '23.6 M, 2048-d']),
     ('ConvNeXtV2-T (Cytoland)', ['init: image-to-image model', '27.9 M, 768-d']),
     ('ConvNeXt-S', ['self-supervised init', '49.5 M, 768-d'])]
for i, (lab, sub) in enumerate(M):
    d.box(lab, 53, 13 + i * 15.5, 68, 13, sub=sub, parent=g_m, id=f'm{i}', color='apricot' if i == 2 else None)
g_h = d.group('Heads (each member)', 131, 14, 40, 56, id='heads')
reg = d.box('Linear', 134, 22, 34, 18, sub=['13 regression targets', 'masked MSE'], parent=g_h, id='reg')
d.box('Linear (aux)', 134, 46, 34, 17, sub=['3 / 10 classes', 'CE x 0.5, train only'], parent=g_h, id='cls')
avg = d.box('Mean', 177, 28, 14, 14, sub=['over 4'], fill='solid', id='avg')
out = d.box('Output', 195, 22, 18, 26, sub=['11 scored', 'targets', 'per cell'], fill='solid', id='out')
d.arrow(g_in, g_m, route='h', ta=0.5, tb=0.5)
d.arrow(g_m, g_h, route='h', ta=0.5, tb=0.5)
d.arrow(reg, avg, route='h', ta=0.5, tb=0.5)
d.arrow(avg, out, route='h')
print(d.save(name='fig6_architecture', sub='overview', root=R,
             notes='Best measured model (N084, GOAL .6911; registered best N053 = first three members, .6862). Each member is an image '
                   'encoder + linear heads, trained separately (10 epochs, AdamW lr 5e-4, batch 256); test predictions of the members are '
                   'averaged. Input: one cell, 24 channels (the same cell imaged under 23 LED illumination angles + a computed phase image), '
                   '128 x 128 px; 2x zoom = centre 64 px upsampled to 128. Self-supervised init: 2 augmented views, invariance + SIGReg '
                   'Gaussian regulariser, 40 epochs (ResNet-50) / 60 (draw 1) and 120 (draw 0) epochs (ConvNeXt-S) on unlabelled cells of the training split only '
                   '(held-out split excluded); the SSL stem has 48 inputs (24 channels + 24 channel-present masks). Cytoland = ConvNeXtV2-T '
                   'encoder of a published microscopy image-to-image model (label-free -> fluorescence), 1-channel stem repeated to 24, '
                   'learned per-channel gate, encoder lr x 0.1. Targets: 13 per-cell regression targets (8 single proteins, 4 pooled, 1 '
                   'background signal), but each training cell has only the target of its own slide measured -> masked MSE over observed '
                   'targets; 11 are scored (GOAL = Pearson r, train on one blood draw, test on the other, mean of both directions). '
                   'Aux heads: 3- and 10-class cell-type cross-entropy (weight 0.5), not used at test time. ResNet-50 members were trained '
                   'on a label variant (cf150), the others on v10 labels; all scored on v10.'))

# ---------------------------------------------------------------- Figure 7: architectures compared
n053 = pd.read_csv(res('N053', 'plot_n053_single.csv')); n083 = pd.read_csv(res('N083', 'plot_goal.csv'))
n078 = pd.read_csv(res('N078', 'plot_goal.csv'))
v = lambda df, arm: df[df.arm == arm].sort_values('seed').value.values
single = [('ResNet-50\nSSL', v(n083, 'control'), SKY),
          ('ResNet-50\nSSL 4x longer', v(n078, 'r50_160'), SKY),
          ('ConvNeXt-S\nSSL', v(n083, 'cnxs60'), SKY),
          ('ConvNeXtV2-T\nCytoland', v(n053, 'Cytoland'), APR),
          ('ConvNeXtV2-T\nCytoland, full lr', v(n053, 'Cytoland full lr'), APR),
          ('ViT-B\nDinoBloom', v(n053, 'DinoBloom-B'), ROSE),
          ('ViT-B\nDINOv3', v(n053, 'DINOv3-B'), ROSE),
          ('ViT-S\nDINOv3', v(n053, 'DINOv3-S'), ROSE)]
e = pd.read_csv(res('N053', 'n053_scores.csv')).set_index('model').goal
e84 = pd.read_csv(res('N084', 'n084_scores.csv')).set_index('set').GOAL
B3 = 'n048_v9_base+n048_v9_zoom2'
ens = [('ResNet-50\nalone', e['n051_v10_base'], GREY),
       ('ResNet-50\nfull + zoom', e[B3], SKY),
       ('+ ViT-S\nDINOv3', e[B3 + '+n053_dinov3_s'], ROSE),
       ('+ ViT-B\nDINOv3', e[B3 + '+n053_dinov3_b'], ROSE),
       ('+ ViT-B\nDinoBloom', e[B3 + '+n053_dinobloom_b'], ROSE),
       ('+ ConvNeXtV2-T\nCytoland', e[B3 + '+n053_cytoland'], APR),
       ('+ Cytoland\n+ ConvNeXt-S', e84['E (PRIMARY): N053 ensemble + ConvNeXt-S seed 1'], SKY)]
fig, ax = pfig.figure(1, 2, height_mm=78, width_ratios=[1.15, 1])
names = [s[0] for s in single]
pfig.bar(ax[0], [s[1] for s in single], [s[2] for s in single], names=names)
ax[0].set_ylabel('GOAL (mean Pearson r)'); ax[0].set_ylim(0.62, 0.69)
n = 2; vals = [np.r_[s[1], [np.nan] * (n - len(s[1]))] for s in single]
pfig.stats_vs_ref(ax[0], names[0], paired=True, values=vals)
pfig.bar(ax[1], [np.array([x[1]]) for x in ens], [x[2] for x in ens], names=[x[0] for x in ens])
ax[1].set_ylabel('GOAL (mean Pearson r)'); ax[1].set_ylim(0.66, 0.70)
pfig.no_stats(ax[1], 'one value per ensemble (fixed member seeds); noise sd .0027 from seed pairs')
for a in ax:
    for t in a.get_xticklabels(): t.set_rotation(35); t.set_ha('right'); t.set_rotation_mode('anchor'); t.set_fontsize(6)
pfig.save(fig, name='fig7_architectures', sub='overview', root=R,
          source=[res('N083', 'plot_goal.csv'), res('N078', 'plot_goal.csv'), res('N053', 'plot_n053_single.csv'), res('N053', 'n053_scores.csv'),
                  res('N084', 'n084_scores.csv')],
          notes='Same task, data split, labels (v10) and fine-tuning recipe for every encoder (10 epochs, linear heads, masked MSE). '
                'a single encoders (rings = seeds; paired t vs ResNet-50 SSL where 2 seeds, Holm): blue = initialised by our own '
                'self-supervised pretraining on unlabelled cells of the training split; orange = initialised from a published microscopy '
                'image-to-image model (Cytoland, encoder lr x 0.1; full lr = x 1); red = ViTs pretrained on natural images (DINOv3) or '
                'blood-smear photos (DinoBloom), 24-channel patch embedding adapted, 128 -> 256/224 px. Params: ResNet-50 23.6 M, '
                'ConvNeXt-S 49.5 M, ConvNeXtV2-T 27.9 M, ViT-S 23.7 M, ViT-B 85.7-89.8 M. b ensembles = mean of member predictions '
                '(seed 0 members; ResNet-50 full + 2x-zoom pair from N048; third member chosen among 4 encoders = N053 record .6862, '
                'replicated .6864 with a fresh seed; 4th member ConvNeXt-S seed 1 pre-registered in N084 = .6911, below the .6916 '
                'improvement threshold). Seed noise sd .0027.')
print('done')
