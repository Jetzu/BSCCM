# N028 · Virtual Wright/Giemsa staining from label-free LED images (paired LF->GS translation on 15.7k registered pairs): model, held-out-slide evaluation, downstream agreement with the real-Giemsa student

type: exp · date: 2026-11-10 · status: done
goal_effect: n/a · scale: module · value: - · why: measurement

## Goal
Virtual Wright/Giemsa staining from label-free LED images (a stated project aim), trained on the 15.7k registered LF<->GS pairs; later a data engine (synthetic Giemsa for the ~350k marker-labelled cells).

## Plan
Alignment probe (claude_alignment_probe.py): GS centre ~70 % <-> 128 px LF, brightfield-GS grey peak NCC .78, residual shift median 2.5 px. Codex WRITER (codex/, brief.md) built register/train/eval (UNet, L1 + blurred L1 + DinoBloom token loss, PatchGAN off). Claude red-team (redteam_claude_N028.md): B1 BLOCKER reproduced first-hand (40 cells: whitened phase correlation agrees with brute-force NCC optimum 5/40, median error 6.1 px, zero shift better 30/40) -> exhaustive unwhitened ZNCC; M1 NCC gate cannot detect mismatched pairs -> gate off, N003 area-suspect pairs excluded from TRAINING only; M2 wandb dir override -> removed (sbatch WANDB_DIR + cron); m5 non-finite steps skipped. Landed in scripts/vs_{common,register,train,eval,smoke}.py. Minor m1/m2/m3/m4/m6/m7 open.
Run (FOV holdout): register CPU 1295508 (48 cores) -> train GPU 1295509 (40 ep, bs 64, 1 H100) -> eval 1295510; outputs /hnvme/workspace/v181bb10-bsccm/runs/n028_vs_fov/. Downstream check: run the TT6 Giemsa student on generated vs real images of held-out cells (autofl truth within slide).

## Changes
- scripts/vs_common.py, vs_register.py, vs_train.py, vs_eval.py, vs_smoke.py (Codex writer, landed with red-team fixes B1/M1/M2/m5); scripts/vs_downstream.py (Claude; student on virtual vs real stains).

## Evidence
- Registration (job 1295508, 19 s, 48 cores): global scale .67, median residual shift 2.24 px, 99.6 % of pairs accepted (only search-boundary rejects).
- Training (job 1295509, 40 epochs, bs 64, 10.5 min on 1 H100): held-out L1 .025, PSNR 27.7, SSIM .88 (train_last_evals.jsonl).
- Evaluation (job 1295510; vs_eval_metrics.json, montage_fov.png; 1,612 held-out split-2 cells): PSNR 27.7 (b0 27.9 / b1 27.3), SSIM .881, DinoBloom-B cosine .764; exact-cell retrieval among 1000 (chance .001): generated->real top-1 .034 / top-5 .117, real->generated .084 / .215; colour mean/std match real (e.g. R .835/.185 vs .838/.185). Visually: hue (purple granulocytes / blue lymphocytes), pink granules and shape reproduced; fine granule texture smoothed.
- Downstream (job 1295574; downstream_tt6s0.json; TT6 student n025_tt6_s0, 8-view TTA): autofl truth within slide (clean v4c truth) virtual .465 / .609 vs real registered crop .568 / .820 vs real full field .582 / .843 -> the virtual stain keeps 74-82 % of the real-stain truth correlation. Student agreement virtual vs real (same crop): CD45 .95, CD16 .93, CD14 .90, CD3 .90, grp CD3/19/56 .86, HLA-DR .75, CD56 .68, CD19 .53, CD123 .49 (12-marker mean .79; real crop vs full field .96).

## Outcome
WORKS (first model, FOV holdout, 1 seed): label-free -> virtual Wright staining reproduces cell-specific appearance (retrieval 34-84x chance) and the lineage-level information a Giemsa reader uses (student agreement .90-.95 for CD45/CD16/CD14/CD3) but loses fine detail (CD123/CD19 ~.5; autofl truth 74-82 % of real). Not a GOAL attempt. Downstream script not yet cross-reviewed -> [partial].
<!-- on close: bin/effect.sh N028 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N028-slug/ + external absolute paths -->

## Next
Batch-held-out virtual staining (train on one slide, generate the other); 256 px output and/or PatchGAN for granule texture (check hallucination via downstream agreement); data engine: generate virtual Giemsa for marker-labelled cells and train GS->MK directly, validated on real GS autofl + clinical contrasts.
