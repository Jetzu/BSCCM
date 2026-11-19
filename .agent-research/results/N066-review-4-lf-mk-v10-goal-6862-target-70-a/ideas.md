# N066 ideas: synthesis (reports: agents/{outsider,assumption_breaker,lit_labelfree,lit_labelnoise,codex_ideas,codex_diag_redteam}.md)

Verified first-hand (lit top claims): Wei et al. ICML 2023 "label separation is preferred over label aggregation when the noise rates
are high" (arxiv 2206.07181); Noise2Noise: zero-mean target noise leaves the L2 optimum unchanged, costs only variance (arxiv 1803.04189);
Pinkard et al. NeurIPS 2025 (arxiv 2405.20559) exists [BSCCM per-marker content: partial, not checked in full text].
Reproduced first-hand: Codex red-team consensus controls (.6722 / .7005 / .6916); assumption breaker censoring fractions and the epoch
trajectory (rise to ep 8, flat 8-10); outsider C1 (r .18). Assumption-breaker numbers not re-run: FOV-bootstrap SD of the GOAL level
.0089, slide-standardisation +.0004, cross-output +.001 -> [unverified], none changes the plan.

Ranked by P(success) x gain / cost (gain = expected GOAL delta single model, before ensembling):
| # | idea | source | scale | P | gain | cost | decision |
|---|---|---|---|---|---|---|---|
| 1 | consensus (v10 + local-bg) TRAINING targets, eval v10 | diagnosis, Codex 1 | module | .35 | +.003-.008 | 1.6 GPU-h | N067 (queued) |
| 2 | exact-symmetry aug (anti-diagonal + LED swap) + 2-view TTA | outsider 2, own geometry test | module | .3 | +.002-.006 | 3.6 GPU-h | N068 (queued) |
| 3 | censored labels as upper bounds + 20 epochs | assumption breaker I2 | module | .25 | +.000-.006 | 3.2 GPU-h | N073 (queued) |
| 4 | dense privileged supervision: auxiliary decoder predicts the cell's 6-ch fluorescence crop (local bg, panel-conditioned), scalar v10 heads unchanged | lit_labelfree 2, Codex 4, N059 parked | paradigm | .2 | +.002-.008 | ~3 GPU-h + 1.5 d build | N074 (Codex write) |
| 5 | one head per label version (separation instead of aggregation) | lit_labelnoise I1 (Wei et al.) | module | .3 | +.003-.008 | 1 GPU-h | after N067 (needs its labels) |
| 6 | precision-weighted targets, 3 background models via triple collocation | lit_labelfree 1 | module | .25 | +.003-.008 | CPU + 2 GPU-h | after N067 |
| 7 | residual-correlation check of v10- vs consensus-trained members -> label-diverse ensemble | lit_labelnoise I2 | knob | - | +.002-.006 ens | 0 GPU-h | on N067 outputs |
| 8 | rescore frozen members on held-out-draw splits 0+1 (10x cells) | assumption breaker I0 | analysis | - | diagnostic | .5 GPU-h | TODO (needs eval-only mode) |
| 9 | b01 recipe audit: w_cls 0 / drop binary grp:CD123 target (class losses act only in b01) | assumption breaker I1 | knob | .3 | +-.005 b01 | 1.2 GPU-h | parked |
| 10 | morphology micro-cohort supervision | Codex 3 | paradigm | .15 | +.002-.006 | needs embeddings | parked |
| 11 | assay-conditioned forward model of 6 raw channels | Codex 5 | paradigm | .15 | +.002-.007 | 3-5 d | parked |
| 12 | raw-space negative-population background correction of training labels (dsb/ADTnorm analogue) | lit_labelnoise I3 | module | .2 | small | 1 d | parked |
| 13 | FOV/cell reliability weights from background disagreement | Codex 2, lit I4, lit_labelfree 3 | module | .2 | +.001-.004 | CPU check first | parked (CPU check after N067 labels) |
| 14 | ensemble-teacher soft labels on all training-draw cells | outsider 1 | - | - | - | - | declined-ish: = N040 noisy student (+.0037, round 2 worse) |

Chosen portfolio (all run concurrently after the 2026-11-19 cluster downtime; total ~11.4 GPU-h < 25): N067 (module), N068 (module),
N073 (module, bundles a schedule knob), N074 (paradigm: dense privileged supervision). Follow-ups conditional on N067: #5, #6, #7.
Unchosen ideas parked in notebook/TODO.md.
Honest expectation (lit + red-team): each lever is a few thousandths; the registered gap (.6862 -> .70) is mostly evaluation-label
error that no inductive model reaches. A combination of 2-3 winners rebuilt as an ensemble could pass .6916 and approach .70.
