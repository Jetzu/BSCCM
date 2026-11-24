"""BSCCM project overview figures (2026-11-24), house style via pfig. Run with ~/miniforge3/envs/triad/bin/python.
Inputs: figures/overview/data/*.csv (make_data.py) + plot-ready CSVs in .agent-research/results/N###-*/.
Outputs: figures/overview/{fig1_progress,fig2_attempts,fig3_gap,fig4_multimarker,fig5_giemsa}.{png,svg,json}"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/.claude/skills/paper-figures')); import pfig
R = '/home/hpc/v181bb/v181bb10/BSCCM'; RES = R + '/.agent-research/results'; D = R + '/figures/overview/data'
os.chdir(R)
SKY, SEA, SAGE, APR, ROSE, LIL = pfig.PALETTE; GREY = pfig.BASELINE
res = lambda n, f: glob.glob(f'{RES}/{n}-*/{f}')[0]


def deltas(path, ctrl, arms, rename):
    """plot_goal.csv (arm, value, seed) -> per-arm GOAL delta vs the seed-matched control + raw values for paired tests."""
    df = pd.read_csv(path); c = df[df.arm == ctrl].set_index('seed').value
    out = {}
    for a in arms:
        v = df[df.arm == a].set_index('seed').value
        s = [x for x in v.index if x in c.index]
        out[rename[a]] = (np.array([v[x] - c[x] for x in s]), np.array([v[x] for x in s]), np.array([c[x] for x in s]))
    return out


def delta_panel(ax, groups, color, ylabel):
    """Bar of GOAL deltas vs control (control = zero bar), paired test on the absolute values per seed."""
    names = ['control'] + list(groups); ctrl_vals = None
    dvals = [np.zeros(max(len(g[0]) for g in groups.values()))] + [g[0] for g in groups.values()]
    pfig.bar(ax, dvals, [GREY] + [color] * len(groups), names=names)
    ax.set_ylabel(ylabel); pfig.threshold(ax, y=0)
    testable = [g for g in groups.values() if len(g[0]) >= 2]
    if testable:
        n = max(len(g[0]) for g in groups.values())
        ctrl = next(g[2] for g in groups.values() if len(g[2]) == n)
        assert all(np.allclose(g[2], ctrl) for g in groups.values() if len(g[2]) == n), 'panel mixes different controls'
        assert all(np.allclose(g[2], ctrl[:len(g[2])]) for g in groups.values()), 'single-seed arm not on the seed-0 control'
        vals = [ctrl] + [g[1] if len(g[1]) == n else np.full(n, np.nan) for g in groups.values()]
        pfig.stats_vs_ref(ax, 'control', paired=True, values=vals)
    else:
        pfig.no_stats(ax, 'one seed per arm')


# ---------------------------------------------------------------- Figure 1: progress of the GOAL over the project
pr = pd.read_csv(f'{D}/progress.csv')
fig, ax = pfig.figure(1, 2, height_mm=62)
ERA = {'v4 labels': SAGE, 'v5 labels': SEA, 'v10 labels': SKY}
for k, kind in enumerate(('single model', 'multi-network')):
    a = ax[k]
    for era, col in ERA.items():
        s = pr[(pr.kind == kind) & (pr.era == era)].sort_values('n')
        if len(s) == 0: continue
        pfig.scatter(a, s.n.values, s.value.values, col, label=era if k == 0 else None, s=14)
        best = np.maximum.accumulate(s.value.values); a.step(s.n.values, best, where='post', color=col, lw=0.9, zorder=2)
    pfig.threshold(a, y=0.70)
    a.set_xlabel('Notebook item'); a.set_ylabel('GOAL (mean Pearson r)')
ax[0].legend(frameon=False, fontsize=6, loc='lower right')
ax[1].set_ylim(ax[0].get_ylim())
pfig.save(fig, name='fig1_progress', sub='overview', source=f'{D}/progress.csv',
          notes='Hand-checked list C in make_data.py (every attempt that measured the common-11 GOAL itself). a best single-network arm per '
                'item (seed mean where several seeds; per-direction encoder choice only where pre-registered, N084 .6744; N083\'s post-hoc '
                'best-per-direction .6753 not shown, its fixed recipe .6681 is); b each item\'s PRIMARY multi-network result (averaged '
                'ensemble; N063 .6784 = composite of two networks). Secondary / descriptive combinations not shown: N065 planned 6-way '
                '.6929 (descriptive secondary), N055 .6912, N083 .6907, N063 .6902, N065 composites .6644-.6830. GOAL record = N053 .6862 '
                '(third member picked among 4, replicated with a fresh seed .6864); improvement threshold .6916 (+2 x noise); N084, the '
                'pre-registered confirmation, .6911 below it. Colour = label version the value was scored on (v5 N023-N050; v10 N051+). '
                'N048 was measured on frozen v5 (.6522 / .6456); its v10 re-score .6804 sits at N051 next to N051\'s v10 retrain (.6788). '
                'Re-bases are not progress. Omitted: v4 era (12-marker metric on v4 labels) and the N082 half-label diagnostics. Step line = '
                'best so far within a label version; dashed line = target .70.')

# ---------------------------------------------------------------- Figure 2: what was tried (single models, delta vs seed-matched control)
heads = deltas(res('N060', 'plot_n060_single.csv'), 'r50 control', ['hlgauss', 'twopart', 'panel', 'pearson'],
               {'hlgauss': 'HL-Gauss head', 'twopart': 'two-part head', 'panel': 'panel head', 'pearson': 'Pearson loss'})
losses = deltas(res('N060', 'plot_n060_single.csv'), 'r50 control', ['fovre', 'pair', 'autofl0', 'sampler'],
                {'fovre': 'FOV offsets', 'pair': 'pair loss', 'autofl0': 'no autofl', 'sampler': 'FOV sampler'})
labels = {}
for n, arm, nm in (('N067', 'cons_N067', 'consensus labels'), ('N075', 'sep_N075', 'label separation'), ('N073', 'cens20_N073', 'censored + 20 ep'),
                   ('N074', 'dense', 'fluor. image aux'), ('N076', 'ftall', 'multi-marker FT'), ('N068', 'diag_N068', 'symmetry aug')):
    labels.update(deltas(res(n, 'plot_goal.csv'), 'control', [arm], {arm: nm}))
enc = {}
enc.update(deltas(res('N083', 'plot_goal.csv'), 'control', ['cnxs60'], {'cnxs60': 'ConvNeXt-S'}))
enc.update(deltas(res('N078', 'plot_goal.csv'), 'control', ['r50_160'], {'r50_160': 'r50, 4x SSL'}))
enc.update(deltas(res('N081', 'plot_goal.csv'), 'control', ['ext'], {'ext': '+ ImageStream'}))
n053 = pd.read_csv(res('N053', 'plot_n053_single.csv')); c83 = pd.read_csv(res('N083', 'plot_goal.csv'))
cseed = c83[c83.arm == 'control'].set_index('seed').value   # n051_v10_base seeds 0/1 (= N053's 'r50 SSL (N051)' seed 0 .6730)
assert abs(cseed[0] - n053[n053.arm == 'r50 SSL (N051)'].value.iloc[0]) < 1e-4
for arm, nm in (('Cytoland', 'Cytoland'), ('Cytoland full lr', 'Cytoland, full lr'), ('DINOv3-B', 'DINOv3-B'), ('DINOv3-S', 'DINOv3-S'),
                ('DinoBloom-B', 'DinoBloom-B')):
    r = n053[n053.arm == arm].sort_values('seed'); c = cseed.loc[r.seed.values].values
    enc[nm] = (r.value.values - c, r.value.values, c)
fig, ax = pfig.figure(2, 2, height_mm=128)
delta_panel(ax[0, 0], heads, SKY, 'GOAL change vs control')
delta_panel(ax[0, 1], losses, SEA, 'GOAL change vs control')
delta_panel(ax[1, 0], labels, APR, 'GOAL change vs control')
enc_main = {k: enc[k] for k in ('ConvNeXt-S', 'Cytoland', 'Cytoland, full lr', 'DINOv3-B', 'DINOv3-S', 'DinoBloom-B', 'r50, 4x SSL', '+ ImageStream')}
delta_panel(ax[1, 1], enc_main, LIL, 'GOAL change vs control')
for a in ax.ravel():
    for t in a.get_xticklabels(): t.set_rotation(30); t.set_ha('right'); t.set_rotation_mode('anchor')
pfig.save(fig, name='fig2_attempts', sub='overview', source=[res('N060', 'plot_n060_single.csv'), res('N053', 'plot_n053_single.csv')] +
          [res(n, 'plot_goal.csv') for n in ('N067', 'N075', 'N073', 'N074', 'N076', 'N068', 'N083', 'N078', 'N081')],
          notes='Single-model GOAL change vs the seed-matched ResNet-50 SSL control (rings = seeds). a output heads/losses (N060-N062), '
                'b loss/sampling variants, c label and supervision changes (N067-N076), d encoders, pretraining and external data '
                '(N053 all five arms, N078, N081, N083; single-seed arms vs the seed-0 control, no test). Control = n051_v10_base seeds 0/1 '
                '(.6730/.6683) in every panel (asserted). Paired t-tests on GOAL per seed (= test of the plotted differences), Holm within panel.')

# ---------------------------------------------------------------- Figure 3: where the gap lives
pt = pd.read_csv(f'{D}/per_target.csv')
order_m = ['ResNet-50 (seeds 0, 1)', 'ConvNeXt-S (seeds 0, 1)', 'Ensemble N053', 'Ensemble N084']
order_t = ['CD16', 'CD14', 'pool CD16', 'pool CD3/CD19/CD56', 'CD45', 'CD3', 'pool CD45', 'CD123', 'CD56', 'HLA-DR', 'CD19']
Mx = pt.groupby(['model', 'target']).r.mean().unstack().loc[order_m, order_t].values
Md = pt[pt.model == 'Ensemble N084'].pivot(index='direction', columns='target', values='r').loc[['b10', 'b01'], order_t].values
fig, ax = pfig.figure(2, 2, height_mm=132, height_ratios=[1.15, 1])
pfig.heatmap(ax[0, 0], Mx, order_m, order_t, vmin=0.3, vmax=0.9, label='Pearson r')
pfig.heatmap(ax[0, 1], Md, ['draw 1 -> 0', 'draw 0 -> 1'], order_t, vmin=0.3, vmax=0.9, label='Pearson r')
lc = pd.read_csv(res('N082', 'plot_goal.csv'))
lmap = {'half10': '50 % labels, 10 ep', 'half20': '50 % labels, 20 ep', 'control': '100 % labels, 10 ep'}
pfig.bar(ax[1, 0], [lc[lc.arm == k].value.values for k in lmap], [SAGE, SEA, GREY], names=list(lmap.values()))
ax[1, 0].set_ylabel('GOAL (mean Pearson r)'); ax[1, 0].set_ylim(0.6, 0.69)
pfig.no_stats(ax[1, 0], 'one seed per arm (N082 pre-registered single-seed reading)')
rf = pd.read_csv(f'{RES}/' + os.path.basename(glob.glob(f'{RES}/N064-*')[0]) + '/full/reconstruction_fidelity.csv')
rf = rf[rf.role == 'eval']; rf['t'] = rf.target.str.replace('single:', '').str.replace('grp:', 'pool ')
rel = [rf[rf.t == t].r_reconstructed_vs_v10.values for t in order_t]
pfig.bar(ax[1, 1], rel, [SKY] * len(order_t), names=order_t)
ax[1, 1].set_ylabel('Label agreement (r)')
for t in ax[1, 1].get_xticklabels(): t.set_rotation(30); t.set_ha('right'); t.set_rotation_mode('anchor')
for t in ax[1, 0].get_xticklabels(): t.set_rotation(20); t.set_ha('right'); t.set_rotation_mode('anchor')
pfig.no_stats(ax[1, 1], 'descriptive per-target agreement (two directions as rings)')
pfig.save(fig, name='fig3_gap', sub='overview', source=[f'{D}/per_target.csv', res('N082', 'plot_goal.csv'), 'N064 full/reconstruction_fidelity.csv'],
          notes='a per-target r (mean of both draw directions; single models = mean r over the same seeds 0 and 1: n051_v10_base(_s1), '
                'n078_ft_cnxs_60_s0 / n083_cnxs60_s1; N053 = base + zoom2 + Cytoland; N084 = N053 + ConvNeXt-S seed 1); b N084 ensemble per '
                'direction (draw 0 -> 1 is the weak direction); c step-matched learning curve (N082, seed 0): labels .0155 > steps .0115; '
                'd agreement of v10 labels with an independent local-background label version on the held-out cells (N064, rings = '
                'directions). Low agreement flags unstable labels (CD19); high agreement does not show the labels are correct (CD123 '
                'agrees .87-.92 yet N084 predicts it at .54 on draw 0 -> 1), so absolute label error is not identified.')

# ---------------------------------------------------------------- Figure 4: multi-marker decoding
mm = pd.read_csv(f'{D}/mm3.csv')
e80 = pd.read_csv(res('N080', 'n080_eval_marker.csv')); pairs = pd.read_csv(res('N080', 'n080_eval_pairs.csv')); wt = pd.read_csv(res('N080', 'n080_eval_withintype.csv'))
fig, ax = pfig.figure(2, 3, height_mm=125)
pfig.bar(ax[0, 0], [mm[mm.arm == a].MM3.values for a in ('ResNet-50 (control)', 'ConvNeXt-S')], [GREY, LIL], names=['ResNet-50', 'ConvNeXt-S'])
ax[0, 0].set_ylabel('Pooled-panel decoding (MM3, r)'); ax[0, 0].set_ylim(0.70, 0.79)
pfig.stats_vs_ref(ax[0, 0], 'ResNet-50', paired=True)
REG = {'full:model': 'all 9 markers', 'single:model': 'one marker/cell', 'single_pooled:model': '+ pooled'}
macro = e80.groupby(['arm', 'test_donor']).r_B.mean().unstack()
oracle = e80[e80.arm == 'full:model'].groupby('test_donor').r_type_oracle.mean().values
g = [macro.loc[k].values for k in REG] + [oracle]
pfig.bar(ax[0, 1], g, [SKY, APR, ROSE, GREY], names=list(REG.values()) + ['cell-type oracle'])
ax[0, 1].set_ylabel('Marker decoding, held-out donor (r)'); pfig.stats_vs_ref(ax[0, 1], 'one marker/cell', paired=True)
MK = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']
pm = e80.groupby(['arm', 'marker']).r_B.mean().unstack()
orc = e80[e80.arm == 'full:model'].groupby('marker').r_type_oracle.mean()
Mm = np.vstack([pm.loc[k, MK].values for k in REG] + [orc[MK].values])
pfig.heatmap(ax[0, 2], Mm, list(REG.values()) + ['cell-type oracle'], MK, vmin=0, vmax=1, label='Pearson r')
PR = [('T', 'CD4-CD8', 'CD4-CD8 (T)'), ('lymph', 'CD3-CD19', 'CD3-CD19'), ('lymph', 'CD3-CD56', 'CD3-CD56'), ('lymph', 'CD19-CD56', 'CD19-CD56')]
sel = lambda arm, grp, pr: pairs[(pairs.arm == arm) & (pairs.group == grp) & (pairs['pair'] == pr)].sort_values('test_donor')
Mp = np.vstack([[sel('full:model', gq, pq).r_true.mean() for gq, pq, _ in PR]] + [[sel(k, gq, pq).r_pred.mean() for gq, pq, _ in PR] for k in REG])
pfig.heatmap(ax[1, 0], Mp, ['measured'] + list(REG.values()), [lab for _, _, lab in PR], cmap=pfig.cmap('house', kind='diverging'),
             vmin=-1, vmax=1, label='Within-population r')
import json
RUNS = '/hnvme/workspace/v181bb10-bsccm/runs'
reg_run = {'full:model': 'full', 'single:model': 'single', 'single_pooled:model': 'single_pooled'}
err = {k: np.array([np.mean([abs(v['predicted_minus_true']) for v in json.load(open(f'{RUNS}/n080_{reg_run[k]}_d{d}/metrics.json'))['model']
                             ['pairwise_correlation_bias'].values()]) for d in (0, 1, 2)]) for k in REG}
pfig.bar(ax[1, 1], [err[k] for k in REG], [SKY, APR, ROSE], names=list(REG.values()))
ax[1, 1].set_ylabel('Pair-correlation error, 36 pairs (|r|)'); pfig.stats_vs_ref(ax[1, 1], 'one marker/cell', paired=True)
wtm = wt.groupby(['arm', 'test_donor']).r_within.mean().unstack()
pfig.bar(ax[1, 2], [wtm.loc[k].values for k in REG], [SKY, APR, ROSE], names=list(REG.values()))
ax[1, 2].set_ylabel('Within-cell-type r'); pfig.stats_vs_ref(ax[1, 2], 'one marker/cell', paired=True)
for a in (ax[0, 0], ax[0, 1], ax[1, 1], ax[1, 2]):
    for t in a.get_xticklabels(): t.set_rotation(25); t.set_ha('right'); t.set_rotation_mode('anchor')
pfig.save(fig, name='fig4_multimarker', sub='overview', source=[f'{D}/mm3.csv', res('N080', 'n080_eval_marker.csv'), res('N080', 'n080_eval_pairs.csv'), res('N080', 'n080_eval_withintype.csv')],
          notes='a BSCCM pooled multi-marker panels (MM3 = mean r of grp:CD3/CD19/CD56, grp:CD45, grp:CD16), ConvNeXt-S in-domain SSL vs ResNet-50 '
                '(seeds 0, 1, N083; pre-registered gain +.0135 > bar .007 on both seeds; paired t at n = 2 has little power). b-f S-BIAD452 '
                'joint-decoding sandbox (ImageStream BF/SSC -> 9 markers, each of 3 donors held out once = rings, N080): b mean r over markers '
                'per training regime and a cell-type oracle (training-donor mean profile of each cell\'s true gated type); c per-marker r '
                '(mean over donors); d correlation of marker pairs within T cells (CD4-CD8) or lymphocytes, measured vs predicted (mean over '
                'donors; the 4-pair mean |error| computed per donor is .389 full / .308 single / .391 pooled, e.g. CD3-CD56 donor means favour full '
                'but per-donor errors favour single); e PRE-REGISTERED pair metric: mean |predicted - measured| correlation over all 36 marker pairs '
                'on whole held-out donors (runs/n080_*/metrics.json, per-run scoring; pairs registered, aggregation = mean |bias| chosen here): full .189, single .229, pooled .237. Joint training is '
                'closer on the registered 36-pair metric and keeps CD4-CD8 exclusivity best; within lymphoid populations single-marker training '
                'is as close; no regime differs significantly at 3 donors.')

# ---------------------------------------------------------------- Figure 5: Giemsa -> markers (clinical transfer)
r58 = pd.read_csv(res('N058', 'n058_per_run.csv')); m57 = pd.read_csv(res('N057', 'plot_n057_mll23.csv')); m54 = pd.read_csv(res('N054', 'plot_m1.csv'))
TCH = ['t0', 't1', 't2']   # students share teachers -> one value per teacher (mean over its seeds), tests paired by teacher
t58 = r58.groupby(['arm', 'teacher']).MLL23_15.mean().unstack()[TCH]
m57['teacher'] = m57.seed.str.split('_').str[0]; t57 = m57.groupby(['arm', 'teacher']).value.mean().unstack()[TCH]
fig, ax = pfig.figure(1, 3, height_mm=70, width_ratios=[0.8, 1, 1.6])
pfig.bar(ax[0], [t58.loc[a].values for a in ('T', 'H_matched')], [GREY, SKY], names=['output distillation', 'hub latent'])
ax[0].set_ylabel('MLL23 fine contrasts (AUROC)'); pfig.stats_vs_ref(ax[0], 'output distillation', paired=True)
pfig.bar(ax[1], [t57.loc[a].values for a in ('anchor .5 (TT6c)', 'anchor 2', 'anchor 8')], [GREY, SEA, SEA], names=['anchor .5', 'anchor 2', 'anchor 8'])
ax[1].set_ylabel('MLL23 fine contrasts (AUROC)'); pfig.stats_vs_ref(ax[1], 'anchor .5', paired=True)
arms54 = ['TT6c student (A0)', 'LF hub (A3)', 'direct map', 'two-space + LF flow', 'two-space + GS flow', 'two-space + unpaired GW']
nm54 = ['output distill.', 'LF hub', 'direct map', '2-space + LF flow', '2-space + GS flow', 'unpaired GW']
pfig.bar(ax[2], [m54[m54.arm == a].sort_values('seed').value.values for a in arms54], [GREY, SKY, SAGE, APR, APR, ROSE], names=nm54)
ax[2].set_ylabel('Same-cell autofluorescence (r)'); pfig.stats_vs_ref(ax[2], 'output distill.', paired=True)
for a in ax:
    for t in a.get_xticklabels(): t.set_rotation(25); t.set_ha('right'); t.set_rotation_mode('anchor')
pfig.save(fig, name='fig5_giemsa', sub='overview', source=[res('N058', 'n058_per_run.csv'), res('N057', 'plot_n057_mll23.csv'), res('N054', 'plot_m1.csv')],
          notes='Giemsa -> marker students via the label-free bridge. Rings = teachers (a, b: mean over each teacher\'s 2-3 student seeds) '
                'or held-out slides (c); tests paired by teacher / slide. a hub latent distillation vs output distillation (TT6c) at matched '
                'class anchoring, MLL23 fine contrasts (N058; registered rule: mean gain over teachers .113 > seed floor .074 = higher; per teacher '
                '+.061, +.109, +.169); '
                'b TT6c anchor-strength sweep (N057; registered: anchor 8 gain .059 < floor .074 = no difference); c tri-modal latent designs '
                'incl. the user two-space plan (N054): same-cell autofluorescence r on the 2 held-out slides; unpaired GW alignment fails.')
print('done')
