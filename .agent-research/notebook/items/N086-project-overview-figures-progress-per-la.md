# N086 · Project overview figures: progress per label era, what was tried (single-model deltas), where the gap is, multi-marker decoding, Giemsa bridge

type: writing · date: 2026-11-24 · status: done
goal_effect: n/a · scale: ? · value: void · why: measurement

## Goal
User 11-24: "figures that nicely describe what we have tried so far and all of the different things that we got out of the project".
Five pfig paper figures (house rules), built only from recorded results (no new runs).

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes
- figures/overview/make_data.py (bsccm.sif): data/progress.csv (all INDEX items with a measured GOAL; label era v4 N007-N022, v5 N023-N047
  and N049-N050 (scored before the N051 re-base), v10 N048 and N051+; ensemble items flagged), data/per_target.csv (per-target r, common-11,
  v10 labels, of ResNet-50 control / ConvNeXt-S / N053 ensemble / N053 + ConvNeXt-S), data/mm3.csv (N083 seeds).
- figures/overview/make_figures.py (triad env, pfig): fig1_progress, fig2_attempts, fig3_gap, fig4_multimarker, fig5_giemsa (.png/.svg/.json).

## Evidence
Codex red-team round 1 (.agent-research/results/N086-project-overview-figures-progress-per-la/codex_round1/findings_round1.md): THE CLAIM BREAKS. All objections reproduced first-hand and fixed:
- fig1 used log values: N026/N037/N041/N050 values are ensembles (shown as single); N007 is a 12-marker linear probe; N022 .6306 is
  b1to0-only common-11 v5 (its v4 value .5722); N010/N015/N020/N021 carry the standing best (Giemsa items), not a measurement; N031 .6527,
  N042 .6411, N063 composite .6784 and N050 ConvNeXt-T .6135 were missing; N083 .6753 was a best-per-direction pick (pre-registered 60-ep
  2-seed mean .6681). -> hand-checked list C in make_data.py; v4 era dropped (different metric); N082 diagnostics omitted (stated).
- fig2 d: Cytoland seed 1 was differenced against control seed 0 (.6730 instead of .6683): bar -.0070 vs test -.0046. DINOv3-S .6495 and
  Cytoland full-lr .6464 were missing. -> seed-matched controls, all N053 arms, assertions that a panel uses one control.
- fig3 a: control seed 0 next to ConvNeXt-S seed 1 -> both models = mean per-target r over seeds 0 and 1 (reproduces .6706 / .6681).
  d caption overclaimed ("weak targets = noisy labels"): CD123 agrees .87-.92 between label versions but N084 predicts it .54 (b01).
- fig4 c generalised from CD4-CD8: over all 4 pairs single-marker training has the LOWEST pair-correlation error (.308 vs full .389,
  pooled .391; ns). -> fig4 now shows per-marker r, all 4 pairs and the pair error; caption: joint training keeps CD4-CD8 exclusivity best,
  single-marker training is closer on the lymphoid pairs.
- fig5: Welch tests treated students of the same teacher as independent (a: p .00018 ***). Teacher-level paired tests: a hub .681 vs
  output distillation .568, p .069 ns (registered N058 rule: gain .113 > floor .074 on every teacher); b anchor 8 p .054 / Holm .108 ns
  (registered: no difference); c paired by held-out slide, all ns.
Codex round 2 (.agent-research/results/N086-project-overview-figures-progress-per-la/codex_round2/findings_round2.md): six more objections, all reproduced first-hand and fixed:
- N063 composite counted as single -> new kind 'multi-network' (ensembles + composite); N063 single = component .6766.
- N048 .6804 was N051's v10 re-score; N048 itself measured frozen v5 .6522 ens / .6456 base -> N048 on v5, N051 holds .6804 + its own
  v10 retrain .6788 + base seed 0 .6730 (the .6706 control mean used an N060 seed).
- missing values: N040 single .6529, N047 ensemble .6571, N065 descriptive 6-way .6929 (now shown, captioned post hoc). Rule made
  uniform: best single-network arm per item (N026 .6422, N031 .6442, N037 .6445, N041 .6450, N046 .6510, N055 .6704, N068 .6718, N084 .6744).
- fig4 e used a 4-pair within-population error that favours single (.308 vs .389); the PRE-REGISTERED metric (36 pairs, whole donor,
  /hnvme/workspace/v181bb10-bsccm/runs/n080_full_d0/metrics.json (and n080_single_d*, n080_full_d1/d2)) favours full: .189 / .230 single / .237 pooled (per donor full .198 .200 .170; single .271 .211 .207).
  -> e now shows the registered metric; the 4-pair numbers are in the caption.
- fig4 d caption: donor-mean correlations vs per-donor errors (CD3-CD56) qualified.
- fig5 a caption: registered rule used the MEAN gain .113 > floor .074 (per teacher .061 / .109 / .169), not every teacher.
Final numbers: fig1 best single v10 .6766 (N063 component, seeds 2-3) / .6756 (N060); multi-network: pre-registered best .6862 (N053,
GOAL record), later pre-registered .6899-.6911 below the .6916 bar, post-hoc 6-way .6929 (N065). fig2 all ns. fig3 weakest targets CD19
.35, HLA-DR .47, CD56 .52, CD123 .56. fig4 MM3 .7711 vs .7576; S-BIAD452 macro r .596/.526/.542, oracle .814; registered pair error
.189/.230/.237; within-type .20/.18/.15 (all ns at 3 donors). fig5 a hub .681 vs .568 (teacher-paired p .069; registered rule: higher).
Codex round 3 (.agent-research/results/N086-project-overview-figures-progress-per-la/codex_round3/findings_round3.md): fig4 e reproduced (single .229, rounding fixed), fig5 a caption
matches; fig1 selection: secondary ensembles above the plotted values (N055 .6912, N083 .6907, N063 .6902) and N065 6-way .6929 is a
PLANNED descriptive secondary (not post hoc). Reproduced (N053 item: Cytoland picked among 4 members, replicated .6864). Fixed by an
explicit rule: panel b = each item's PRIMARY multi-network result; all secondary/descriptive combinations listed in the caption with the
record threshold .6916 and N084 .6911. No further data errors found in round 3; no round 4 (remaining fix narrowed a caption claim).


### Extension (user 11-24): architecture + alignment figures for ML readers
- fig6_architecture / fig7_architectures (make_architecture.py): best model diagram (N084 4-member ensemble; dims/params measured
  from timm/bsccm code) and every encoder on the same protocol + ensemble build-up.
- fig8_alignment_crossmodal / fig9_alignment_domain (make_alignment.py): all alignment experiments. A Claude subagent extracted 939
  metric rows with file:line sources -> figures/overview/data/alignment_metrics.csv + alignment_notes.md.
- Codex round 4 (.agent-research/results/N086-project-overview-figures-progress-per-la/codex_round4/): all four figures broke; reproduced first-hand and fixed: targets per slide type and
  aux-label availability (make_targets_v3.py:6-11, v5:18-19), illumination = 23 patterns not angles, fine-tune aug = +-4 px shift only;
  fig7 b build-up now starts from the cf150 full-view member .6742 (was the v10 model .6730); DINOv3 rounding .651/.649; fig7 a caveat
  (lr/resolution/pooling confounded). fig8: within-slide SigLIP R@10 .40/.38 vs cross-slide .02-.04 -> the failure is slide shift, NOT
  missing shared information (caption claim reversed); b = complete pipelines (N054 transductive LF training, GW step on a paired
  encoder); c one draw, transductive features, 6 targets at n=2048. fig9: full-precision PBC AUROCs (pbc_transfer_auroc.csv; DANN .3 =
  .5546), per-teacher post-hoc alignment values from N056 tables/contrasts_all.csv (OT teacher 0 -.038), readout difference noted,
  DANN deltas vs full-precision n013 baseline (+.0025 / -.0015), 'never helped' softened (hub + mean shift PBC4 +.053 > floor .048).
- Inconsistencies found by the extraction agent: CONFIRMED N045 text averages two autofl truths (figures use v5c from v2_summary.csv);
  CONFIRMED N014 baseline .5545 = 4-run mean vs seed-0 .5536 used for pairing. [unverified]: N020:19 / N021:17 in-domain DANN cost
  (.48-.49 vs .54) may compare against N010 cross-slide runs instead of the matched N019 FOV run (.485); N056 vs N058 H_matched
  seed 0 MLL23_15 differ (.632/.653/.634 vs .675/.642/.673); N058 title says 'fine contrasts' but scores MLL23_15 (12 fine + 3 lineage).

## Outcome
DONE: five overview figures (fig1-fig5) after three Codex red-team rounds; every objection reproduced first-hand and fixed. Writing item, no GOAL effect.
<!-- on close: bin/effect.sh N086 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
- /home/hpc/v181bb/v181bb10/BSCCM/figures/overview/{make_data.py,make_figures.py,data/,fig1..fig5.{png,svg,json}}
- architecture figures (user 11-24, ML audience): make_architecture.py -> fig6_architecture (best model diagram, dims/params measured from code), fig7_architectures (all encoders on the same protocol + ensemble build-up); CAPTIONS.md per-panel descriptions
- copy for the user: /home/hpc/v181bb/v181bb10/bsccm_deliverables/figures/overview/
- red-team: .agent-research/results/N086-project-overview-figures-progress-per-la/codex_round1/ .. codex_round4/ (briefs + findings)
<!-- files in .agent-research/results/N086-slug/ + external absolute paths -->

## Next
Refresh fig1/fig4 when N085 is scored (make_data.py list C + make_figures.py).
