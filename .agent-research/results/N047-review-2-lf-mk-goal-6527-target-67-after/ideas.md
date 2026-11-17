# N047 ideas and portfolio (review 2)

Sources: agents/outsider.md, agents/lit_group_noise.md, agents/assumption_breaker.md (+ agents/ab/), agents/codex_ideas.md,
agents/codex_diag_redteam.md; user's dataset notes 2026-11-17 (labelling efficiency 60-99 %, cross-reactivity, single stains do not
test other markers, diagnosis from cell-type proportions, lineage-defining markers).

| # | idea | source | scale | gain (reasoned) | cost | decision |
|---|------|--------|-------|-----------------|------|----------|
| A | Label v9: asinh cofactor from a label-only rule (negative-population noise, flow-cytometry practice) instead of q10; pre-registered QC; retrain the GOAL recipe; score frozen v5 AND v9 | assumption breaker I1 (reproduced) | module | frozen v5: 0 to +.004 (cleaner CD3/CD56 training labels); v9-scored ~.68 = re-base (user) | CPU + ~1.3 GPU-h | N048 |
| B | Noise-aware positive/negative labels: per-marker mixture model -> P(positive) with explicit labelling efficiency / false-positive rate; train a probabilistic head; report AUROC (population-level metric dropped by the user) | user notes, lit (mixture heads), outsider (gating AUROC) | paradigm | GOAL 0 to +.005; new population-level metric | ~1.5 GPU-h | N049 |
| C | (DROPPED by user 2026-11-17: 'doesn't matter for now') Lineage-consistency prior: B/T/NK/monocyte exclusivity (CD19 vs CD3 vs CD56 vs CD14) as a structured head / consistency loss anchored on pooled-panel cells + 10-class gates; also checks pseudo labels | user notes, Codex 2 (repeated-antigen consensus) | module | 0 to +.006 on weak lymphoid targets | ~1.5 GPU-h | dropped (user) |
| D | FOV-context offset head | outsider A, lit 1 | module | refuted: offsets not image-visible (R^2 <= .03) | - | dropped |
| E | k-shot isolation calibration (fit a 13-head readout on k labelled cells of the new isolation) | assumption breaker I2 | protocol | +.016-.022 at k=500 (oracle-free estimate) | 0 GPU-h | user decision (re-scope) |
| F | Within-FOV contrast loss | Codex 3 | knob-module | 0 to +.006; lit says unlikely | 1.6 GPU-h | parked |
| G | Censor contradictory source strata | Codex 4 | module | 0 to +.006 | audit 0 GPU-h | parked (audit later) |
| H | Heterogeneous member: different backbone (ConvNeXt-T, scratch-free SSL-free init) trained on real + N040 pseudo labels -> decorrelated errors (red-team: residual r .97 among members) | lit 3, Codex diag red-team | module | +.003-.008 ensemble | ~1.5 GPU-h | N050 |
| I | Re-scoping: R1 marker info beyond morphology, R2 coverage/abstention, R3 improvement over 10-model reference with CIs | Codex R1-R3 | - | - | - | user decision |

Portfolio: N048 (module, A), N049 (paradigm, B), N050 (module, H); ~4.5 GPU-h total. Re-scoping options for the user: (1) v9 cofactor
re-base (label-curation fix, N023 precedent), (2) k-shot isolation calibration as a protocol variant; target .67 itself was never
confirmed. User 2026-11-17: lineage priors and diagnosis/population metrics do not matter for now.
