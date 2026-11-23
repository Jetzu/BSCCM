# N079 · Review 5: LF->MK v10 GOAL (.6862, target .70) after 5 flat attempts (N067/N068/N073/N075/N076); user re-focus 11-23: multi-marker decoding, external data + encoders allowed

type: review · date: 2026-11-23 · status: done
goal_effect: pending · scale: n/a

## Goal
Project review 5 (3rd without progress): 5 flat attempts N067/N068/N073/N075/N076; user re-focus 2026-11-23 (multi-marker decoding; external data + encoders allowed).

## Plan
Shared assumption (attempts.md): re-processing the SAME BSCCM measurements can add information; none added a new measurement.

## Changes
Diagnosis (diagnosis.md, corrected after the Codex red-team, reproduced): joint per-cell decoding is NOT measurable on BSCCM (pooled
channels ~ cell type: cell-type-only .730 vs composed singles .737 vs head .746, b01); "data-limited" is confounded with optimisation
steps (K033). External: S-BIAD452 verified (9 markers on every cell; packer written + smoke-tested; download in progress).
Agents (4): outsider, Codex ideas, Codex diagnosis red-team (after a credential refresh), lit panel merging (Infinity Flow, Mocking 2023).

## Evidence
Portfolio: N080 joint-decoding sandbox on S-BIAD452 (paradigm), N081 two-stem BSCCM+S-BIAD452 training (module), N082 step-matched
learning curve (knob); pending N074 (dense fluorescence supervision) and N078 (SSL scaling). Parked ideas -> TODO.md.
Re-scoping candidates for the user (3rd review without progress): (a) report joint per-cell decoding on S-BIAD452 as the primary
multi-marker evidence next to the BSCCM GOAL; (b) accept ~.69 as the BSCCM inductive ceiling and lower the target; (c) few-shot new-draw
calibration (N059: .7042 with 200-500 labelled cells/target).

## Outcome
<!-- on close: bin/effect.sh N079 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N079-slug/ + external absolute paths -->

## Next
