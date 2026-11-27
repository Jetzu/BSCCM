# N095 assumption breaker (Claude subagent, 2026-11-27)

I rebuilt E8 from the member code in `n085_score.py`. It reproduces b10 .7066 and b01 .6869. All files cited below are in `/tmp/n095_ab/`, with the scripts as `ana*.py` and the outputs as `ab_*`. The integrator should copy them to `agents/ab_data/` before committing.

## Checks run (CPU, saved predictions only)
| # | check | result | file |
|---|---|---|---|
| 1 | Per-slide offsets (the hypothesis in diagnosis.md): within-slide centring / per-slide affine ORACLE / training-free per-slide z of predictions | GOAL .6981 / .6996 / .6971 (+.0014 / +.0029 / +.0004). **Refuted**: per-slide offsets are not where the gap is. Only the b01 pooled targets span more than one slide | ab_oracles_goal.csv |
| 2 | Censoring | r on uncensored cells .6899 (lower than .6967); a censoring-consistent oracle gives .6976. **Refuted** | ab_oracles_goal.csv |
| 3 | FOV-level label component, corrected by 20 within-slide FOV permutations | +.040 GOAL (b10 +.036, b01 +.044). CD19 +.19/+.18, CD56 +.09/+.09. The effect is real but **appears in both directions**: the b01-only excess is about +.008. A neighbour-brightness context model transferred across directions changes GOAL by −.003 | ab_fov_null_type_goal.csv, ab_fov_null_type.csv, ab_neighbour_context.csv |
| 4 | b01 replicate slides | grp:CD45 scores .579 on rep0 and .884 on rep1. Within-type label SD (lymphocytes/granulocytes/monocytes) is 1.05/.85/1.17 on rep0, .27/.40/.35 on rep1 and .73/.54/.50 on the draw-0 'all' slide. Rep0 also carries a nuisance shared across dyes: the label-only r(grp:CD3/19/56, grp:CD45 \| top 40 % of grp:CD16) is .62 on rep0, .14 on rep1 and .09 on draw 0 | ab_per_slide.csv, ab_replicate_vs_draw0.txt, ab_replicate_dye_corr.txt, ab_rebase_qc.txt |
| 5 | In-distribution model `n039_ceil_fov_v5_mt0` (saw draw-1 split-0 labels; trained on v5 labels, which handicaps it) on the SAME b01 cells, scored on v10 labels | b01 .6963 vs E8 .6869. HLA-DR .580 vs .448; grp:CD45 .708 vs .662 (rep0 .656 vs .579); grp:CD3/19/56 .822 vs .795. The data-matched 128k model still reaches HLA-DR .563. HLA-DR r within lymphocytes: .65 vs .28 | ab_indist_vs_e8.csv, ab_indist_vs_e8.txt |
| 6 | Relations between the 13 outputs | Pooled-with-single head blend: +.0005 GOAL. A 13→1 linear stacker fitted in one direction and applied to the other: −.068 (b01) / −.060 (b10). Adding the CD19 head to HLA-DR in lymphocytes: ≤ +.005 on one target, although the CD19 head tracks HLA-DR within lymphocytes better (.46/.36) than the HLA-DR head (.19/.14) | ab_crosshead.csv, ab_crossdir_stacker.csv, ab_hladr_lymph_crosshead.txt, ab_hladr_cd19_rule.txt |
| 7 | Test-set sampling | FOV-bootstrap SD of GOAL .0092 (b10 .0135, b01 .0118), 3.4× the seed SD. 38 % of resampled test sets put E8 at ≥ .70 | ab_fov_bootstrap.csv |
| 8 | Source-draw size | Cells per draw: 137,961 in draw 0 vs 274,980 in draw 1. Pooled: 10,210 (one 'all' slide) vs 27,074 (two replicates). CD3 6,538 vs 31,332; CD16 9,397 vs 27,857. N082: halving the labels costs .0155 GOAL (b01 −.018) | ab_label_counts.txt; results/N082-*/goal_summary.csv |

Two items diagnosis.md lists as "not yet measured" are now measured. (a) Is in row 1. (b) If HLA-DR, grp:CD45 and grp:CD16 reached their b10 values, GOAL would be ≈ .710 (computed from ab_oracles_per_target.csv).

**Bottom line.** Two things make up the b01 deficit. (a) Draw 1 has its own mapping from image to label, which draw-1 labels teach and draw-0 labels cannot (HLA-DR and the pooled targets). (b) One test slide, draw-1 'all' rep0, has a label artefact that also caps an in-distribution model (.656). Neither is a problem of representation, decoder or ensemble variance. To reach .70, b01 must gain +.0065 (+.0107 for the .7021 bar). The only measured reference is an in-distribution single model at .6963 (one model, trained on v5 labels), and E8 is .0094 below it.

## Assumptions shared by N087/N090/N091/N093/N094: status and cheapest overturning test
- **D1** A better encoder can absorb the draw shift. **Overturned** (#5): a weaker in-distribution model beats E8 by .13 on HLA-DR.
- **D2** The source draws are comparable in size. **False** (#8): b01 has half the data, and its pooled heads learn from one slide.
- **D3** Every test slide is an equally valid yardstick. **Overturned** (#4, a 10-CPU-minute label-only statistic).
- **L1** The per-cell label is a fixed, correct target. Partly false: the FOV part is worth +.040 but sits in both directions (#3), and rep0 carries an artefact (#4).
- **L2** Per-slide offsets matter: **refuted** (#1). **L3** Censoring matters: **refuted** (#2, N073).
- **R1/M1** The gap lies in features, decoder, capacity or ensemble variance. **Refuted** (#5; N087, N090, N094; member residuals correlate .95–.97, K029).
- **O1** The 13 outputs are independent, so co-expression is a free lever. **Refuted post hoc** (#6): the relations change between draws, as in the N096 sign flips. Training-time joint priors are untested on GOAL, but #6 caps the HLA-DR route at about +.0002 GOAL.
- **E1** A gap of .0033 is measurable on this fixed split. **Weak** (#7): it is about one third of a bootstrap SD, and 6 of the 8 members were chosen after their test-draw scores were known (README caveat).
- **E2** The b01 pooled targets come from one population. **False**: one r pools two replicates of different label quality, and rep0 supplies 1,565 of the 2,807 cells.

## Pre-mortem: "Feb 2027, GOAL still < .70"
1. We kept modelling draw 0 better, but the missing b01 gain was a draw-1-specific mapping that only draw-1 labels or more donors carry.
2. The rep0 artefact held grp:CD45 at .58–.66 for every model, and nobody asked the user about a label-only QC re-base.
3. Real gains of +.003–.005 stayed under the bar, or were specific to this split (SD .0092).
4. Ensembles kept being rebuilt after they had saturated (+.002 per member).
5. The FOV-level background error (+.04) stayed invisible in the images, and every fix for it was a re-base.

## Ideas
1. **Label-only replicate QC for the draw-1 'all' slides.** Ask the user first: this is a re-base, not progress. *Mechanism:* mask a slide's pooled labels when its label-only cross-dye r exceeds .3 (rep0 .62, the others .14 and .09), following the K019 precedent. *Why it moves the bottleneck:* it removes a measurement floor that even the in-distribution model hits (.656). *Expected:* on the same E8 predictions, GOAL becomes .7068 (grp:CD45 only) or .7090 (all three pooled targets) (ab_rebase_qc.txt). *Falsifier:* 0 GPU-h; freeze the rule before scoring, and drop the idea if it also flags draw-0 slide 8 or needs a model. *Build:* < 1 h.
2. **Curate the same slide on the training side.** This is inductive, not a re-base: the test labels stay unchanged. In b10, the pooled heads train on rep0 (13,675 of 27,074 pooled cells); drop or down-weight its pooled labels using the same label-only rule. *Moves:* the b10 pooled heads (grp:CD45 b10 is .784). *Expected:* +.001–.004 GOAL, traded against N082's label limit. *Falsifier:* one b10 composite r50 run against the control, about 1.5 GPU-h; stop if grp:CD45 b10 gains less than +.02. *Build:* low (mask a copy of the target file).
3. **Dense pseudo-labels from the E8 teacher on source-draw cells,** re-testing K033 on v10. *Mechanism:* b01 is the half-data direction and N082 found it label-limited. Densifying every source cell to all 11 targets takes CD3 from 6.5k to 138k labels. *Moves:* only the data-limited share of b01, not the draw-specific share (#5), so the gain is modest. *Expected:* single model +.005–.010, ensemble +.002–.004. *Falsifier:* E8 inference on source rows plus one r50 student per direction, about 4–6 GPU-h; stop if the student scores below control .6706 + .0054. *Build:* low (`--pseudo` already exists in scripts/train_lf.py).
4. **Reframe the bar.** User decision, 0 GPU-h. Report GOAL with its FOV-bootstrap 95 % interval (±.018) next to the in-distribution reference. On this split, .70 and .6967 are not distinguishable.

Ideas 2 and 3 respect declined.md: no test-draw labels or images enter training. Ideas 1 and 4 change the yardstick and need the user's explicit OK.
