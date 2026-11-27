"""Figure 10 (N087): decoders, autoencoders and marker priors, for ML readers. Run with ~/miniforge3/envs/triad/bin/python.
Sources: .agent-research/results/N087-*/{decoders,decoders_ae,cluster,sandbox_eval,fullninth_eval}/ and the statistics baseline
(/hnvme/workspace/v181bb10-bsccm/runs/n087/decoders_stats/goal.csv, copied to results N087 decoders_stats/)."""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
R = '/home/hpc/v181bb/v181bb10/BSCCM'; N = glob.glob(R + '/.agent-research/results/N087-*')[0]
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE; GREY = pfig.BASELINE
dec = pd.read_csv(N + '/decoders/goal.csv'); dae = pd.read_csv(N + '/decoders_ae/goal.csv'); dst = pd.read_csv(N + '/decoders_stats/goal.csv')
G = pd.concat([dec, dae, dst]); G['k'] = G['k'].astype(str)
clu = pd.read_csv(N + '/cluster/cluster_with_imgstats.csv'); clu2 = pd.read_csv(N + '/cluster/cluster.csv')
sb = pd.read_csv(N + '/sandbox_eval/n087_eval_36pairs.csv'); fn = pd.read_csv(N + '/fullninth_eval/n087_eval_36pairs.csv')
fig, ax = pfig.figure(2, 2, height_mm=128)
# a: GOAL vs bottleneck rank on frozen features (fine-tuned vs SSL), dashed = model's own head
for fs, col, lab in (('finetuned', SKY, 'fine-tuned ResNet-50'), ('ssl', GREY, 'self-supervised, not fine-tuned')):
    s = G[(G.feature_set == fs) & (G.decoder == 'reduced_rank') & (G.k != 'full')].copy(); s['kk'] = s.k.astype(int); s = s.sort_values('kk')
    pfig.line(ax[0, 0], s.kk.values, s.GOAL.values, col, label=lab)
pfig.threshold(ax[0, 0], y=float(G[(G.feature_set == 'finetuned') & (G.decoder == 'reference')].GOAL.iloc[0]))
pfig.log_axis(ax[0, 0], 'x'); ax[0, 0].set_xlabel('Decoder bottleneck rank k'); ax[0, 0].set_ylabel('GOAL (mean Pearson r)')
ax[0, 0].legend(frameon=False, fontsize=6, loc='lower right')
# b: frozen features + MLP decoder (rings = directions)
FS = [('finetuned', 'fine-tuned\n(own head)', GREY, 'reference'), ('ssl', 'self-supervised', SKY, 'mlp'), ('imgstats', 'image\nstatistics', SAGE, 'mlp'),
      ('ae16', 'autoencoder\n16-d', APR, 'mlp'), ('ae64', 'autoencoder\n64-d', APR, 'mlp'), ('ae256', 'autoencoder\n256-d', APR, 'mlp')]
vals = [G[(G.feature_set == f) & (G.decoder == d)][['b10', 'b01']].values.ravel() for f, _, _, d in FS]
pfig.bar(ax[0, 1], vals, [c for _, _, c, _ in FS], names=[n for _, n, _, _ in FS]); ax[0, 1].set_ylim(.55, .69)
ax[0, 1].set_ylabel('GOAL, frozen features + MLP'); pfig.no_stats(ax[0, 1], 'one model per representation; rings = the 2 directions')
# c: clustering of held-out draw-0 cells vs gated cell types (kNN balanced accuracy)
CL = [('ssl', 'self-supervised', SKY), ('imgstats', 'image\nstatistics', SAGE), ('ae16', 'autoencoder\n16-d', APR), ('ae64', 'autoencoder\n64-d', APR),
      ('ae256', 'autoencoder\n256-d', APR)]
allc = pd.concat([clu, clu2]).drop_duplicates(['feature_set', 'representation'])
nat = allc[allc.representation == 'native'].set_index('feature_set')
pfig.bar(ax[1, 0], [np.array([nat.loc[f, 'knn_balanced_accuracy']]) for f, _, _ in CL], [c for _, _, c in CL], names=[n for _, n, _ in CL])
ax[1, 0].set_ylabel('Cell-type kNN balanced accuracy'); ax[1, 0].set_ylim(.25, .38)
pfig.no_stats(ax[1, 0], 'one value per representation (10,137 cells, 10 classes, 5-fold CV grouped by field of view)')
# d: sandbox co-expression error (registered 36 pairs), rings = held-out donors
def per(df, key):
    return df[df.regime.str.contains(key)].sort_values('test_donor').mean_abs_bias.values
SB = [('n080_single', 'one marker\nper cell', GREY, sb), ('sandbox_linear4', '+ linear\nprior (4)', ROSE, sb), ('sandbox_ae6', '+ autoencoder\nprior (6)', ROSE, sb),
      ('full_ninth', 'all markers,\n1/9 of cells', SEA, fn), ('n080_full', 'all markers,\nall cells', SKY, sb)]
pfig.bar(ax[1, 1], [per(df, k) for k, _, _, df in SB], [c for _, _, c, _ in SB], names=[n for _, n, _, _ in SB])
ax[1, 1].set_ylabel('Co-expression error, 36 pairs (|r|)'); pfig.stats_vs_ref(ax[1, 1], 'one marker\nper cell', paired=True)
for a in ax.ravel()[1:]:
    for t in a.get_xticklabels(): t.set_rotation(30); t.set_ha('right'); t.set_rotation_mode('anchor'); t.set_fontsize(6)
pfig.save(fig, name='fig10_decoders_autoencoders', sub='overview', root=R,
          source=[N + '/decoders/goal.csv', N + '/decoders_ae/goal.csv', N + '/decoders_stats/goal.csv', N + '/cluster/', N + '/sandbox_eval/', N + '/fullninth_eval/'],
          notes='N087. a GOAL of a linear decoder with a rank-k bottleneck on frozen features (dashed = the fine-tuned model\'s own head .6730); '
                'predictions need about 8-12 dimensions. b frozen features with an MLP decoder (fine-tuned = own linear head; rings = the 2 '
                'directions): self-supervised features without fine-tuning (.603) are no better than 144 hand-made per-channel image '
                'statistics (.607); autoencoders .594-.624; fine-tuning .673. c clustering of 10,137 held-out draw-0 cells against 10 gated '
                'cell types (kNN balanced accuracy, chance .10): autoencoders (.342-.346) beat self-supervised features (.306) but '
                'equal image statistics (.354) -> they mostly encode brightness/scatter. d ImageStream sandbox (9 markers per cell, rings = '
                '3 held-out donors): registered co-expression error over 36 marker pairs; learned marker priors (fitted on another '
                'donor\'s joint labels) do not reach all-marker training (autoencoder prior lower than single-marker training on all 3 donors, paired t Holm p < .05, but above the pre-registered .21 bar); all-marker training on 1/9 of the cells (same label count) is '
                'as good as on all cells -> joint measurement, not label count, gives the co-expression.')
print('done')
