# N089 · Release (user 11-24): private GitHub Jetzu/BSCCM (branch bsccm-data-model) + private HF Jetzu77/BSCCM weights-only checkpoints + repo README

type: task · date: 2026-11-24 · status: done
goal_effect: n/a · scale: ? · value: void · why: infra

## Goal
User 11-24: "do we have a github repo?" -> created Jetzu/BSCCM ("upload everything ... make sure not to share anything private");
"do you need a huggingface for the checkpoints?" -> created Jetzu77/BSCCM; "write a nice readme that explains everything".

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Secret scan before the first push (all history): no API keys/tokens/private keys/credential files; scripts/fetch.py reads Dryad
  credentials from environment variables only; no blob > 50 MB; Codex run logs contain no auth tokens. Commit author e-mails are in the
  history (private repo). Remote origin = git@github.com:Jetzu/BSCCM.git; pushed bsccm-data-model (no local main branch exists).
- HF: scripts/export_hf.py (weights-only safetensors + config.json + eval_preds.npz + result.json; no optimizer/rows/RNG) -> uploaded
  86 files / 2.72 GB to private Jetzu77/BSCCM: ensemble_N053 (3 members x 2 directions), ensemble_N084 ConvNeXt-S member, control
  ResNet-50, composite_N063 (panel + twopart_fovre, seeds 2-3), 4 SSL encoders (+ projector). Staging dir deleted after upload.
- README: Codex writer (results/N088-*/readme/round1/) -> Claude fact-check (factcheck_claude.md: 13 errors, 8 overclaims, no private
  data) -> Codex fix round -> Claude verified the key fixes (shortcut caveat, R@10 wording, N084 per-marker values, rebuild dirs, no
  e-mails / mentor names) -> README.md at the repo root (884 lines). Note: items/N004 names the supervisor and mentors (asked the user).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
DONE: GitHub remote + HF checkpoints + README.
<!-- on close: bin/effect.sh N089 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N089-slug/ + external absolute paths -->

## Next
