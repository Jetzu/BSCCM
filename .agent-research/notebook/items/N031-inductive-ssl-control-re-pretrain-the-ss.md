# N031 · Inductive SSL control: re-pretrain the SSL(aug) encoder EXCLUDING the held-out batch (n007 used all rows incl. test-batch images), then base + zoom2 fine-tunes and the multi-scale ensemble on the GOAL (red-team N026)

type: exp · date: 2026-11-10 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal
Red-team N026: the SSL(aug) encoder (n007) was pretrained on all rows incl. the held-out batch's unlabelled images -> every SSL-init GOAL number was transductive. Measure the inductive version.

## Plan
Same SSL(aug) recipe (r50, 40 ep, bs 512, lr 2e-3, lam .05, proj 256) with --exclude-batch 0 (for b1to0) and --exclude-batch 1 (for b0to1); fine-tune base and zoom2 (10 ep, lr 5e-4, v5 labels) from each; ensemble. Jobs .agent-research/results/N031-inductive-ssl-control-re-pretrain-the-ss/jobs.txt (SSL 1295538 1h52, 1295541 0h36).

## Changes

## Evidence
- GOAL common-11 (goal_inductive.csv; plot .agent-research/plots/goal-inductive-ssl_2026-11-10_1245.png): transductive vs inductive SSL: base .6423 vs .6425; zoom2 .6403 vs .6442; multi-scale ensemble .6512 vs .6527 (b10 .6612 / b01 .6443).

## Outcome
CONTROL PASSED: excluding the held-out batch from SSL pretraining does not lower any number (ensemble .6527 >= .6512) -> the SSL benefit is not a test-image leak. GOAL re-based to the inductive protocol, best .6527 (N031). Measurement control (n/a), not an attempt.
<!-- on close: bin/effect.sh N031 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N031-slug/ + external absolute paths -->

## Next
