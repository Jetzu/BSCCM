# Codex WRITE brief (N074, BSCCM). Hat: write. Writable root = this folder (/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N074-dense-privileged-supervision-auxiliary-d/codex) ONLY; repo /home/hpc/v181bb/v181bb10/BSCCM
# is read-only for you; scratch, venvs and caches go to /tmp/n074_codex (never into results/ or /hnvme). No GPU (cluster in maintenance):
# CPU smoke tests only, small (<= 64 cells). Python env: apptainer exec --bind /hnvme/workspace/v181bb10-bsccm
# /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I ... (if apptainer/FUSE fails for you, use a /tmp venv with torch+timm+numpy+pandas).

## ONE task: dense privileged supervision for the LF->marker GOAL harness
Add an OPTIONAL auxiliary decoder to scripts/train_lf.py: from the LF encoder's spatial feature map it predicts the training cell's own
6-channel fluorescence crop (training-draw cells only), conditioned on the slide's antibody panel. The scalar v10 regression heads, the
evaluation and the evaluation labels stay exactly as they are; the decoder is discarded at evaluation. Default OFF; with the flag off the
code path, numerics and checkpoint/config layout must be byte-for-byte the legacy ones (follow the existing _NEW_DEFAULTS / NNEW pattern).

Read first: scripts/train_lf.py (current working copy, includes uncommitted N068/N073 flags --diag-aug/--diag-tta/--drop-led/--censor-loss),
bsccm/data.py (pack layout; fluor.npy is NEVER an LF input), bsccm/gpu_norm.py, bsccm/encoders.py, the N064 Codex pipeline
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N074-dense-privileged-supervision-auxiliary-d/../N064-noise-ceiling-bracket-for-the-v10-goal-p/codex/n064_ceiling.py (fluor pack channel order: native pack indices 5..0 map to
CHANNELS; disk/corner geometry FG = r < 64 px, background = corners), /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N074-dense-privileged-supervision-auxiliary-d/../N066-review-4-lf-mk-v10-goal-6862-target-70-a/{diagnosis.md,ideas.md}
and notebook/GOAL.md, notebook/KNOWLEDGE.md (K027: reading markers from generated fluorescence images was worse -> this is an AUXILIARY
training signal only).

Spec:
1. pack_fluor_aux.py (+ .sbatch for preempt_cpu, 48 cores, preempt-safe via the preempt-run.sh pattern used by
   /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N074-dense-privileged-supervision-auxiliary-d/../N064-noise-ceiling-bracket-for-the-v10-goal-p/codex/n064_cpu.sbatch): ONE output file
   /hnvme/workspace/v181bb10-bsccm/fluor_aux64.npy, float16 (N_all, 6, 64, 64) over ALL 412,941 rows (row = global index), memmap-written
   in chunks (no per-chunk files kept: /hnvme inode cap), plus a small JSON with the transform constants. Per cell and channel: subtract the
   cell's local corner background (median of the crop pixels outside the r=64 disk), 2x2 mean-pool 128->64, asinh(x / s_c) with a per-
   channel, PER-DRAW cofactor s_c estimated on that draw's split-0 unstained cells (robust scale), then standardise per channel per draw on
   split-0 cells. Each draw uses only its own statistics (a direction trains on one draw only). Record and justify any choice you change.
2. train_lf.py flags: --dense-aux PATH (default '' = off), --dense-w float (loss weight, default 0.5 when PATH given), and whatever minimal
   knobs you need (decoder width). Decoder: from the encoder's pre-pool feature map (timm forward_features; pooled features must remain
   identical to the current model['enc'](x) output so the scalar heads see the same h), light upsampling conv decoder to (6, 64, 64), FiLM-
   conditioned on a learned embedding of the cell's antibody condition (bsccm_index.parquet 'antibodies'; include unstained/'all'/each
   single antibody). Must work for --arch resnet50 with --init-ssl (48-channel stem with presence planes) and should not break the
   bsccm.encoders archs (cytoland/dinov3) -- if a non-r50 arch has no spatial map, refuse the flag for it with a clear error.
   Loss: masked MSE per pixel (cell disk r < 64 px at 128 -> r < 32 at 64, plus a lower weight, e.g. .25, outside), averaged, x dense-w,
   added to the training loss only. Training rows only; assert no held-out-draw row ever reaches the decoder loss. Diag augmentation
   (--diag-aug) must transform the fluorescence target consistently (anti-diagonal reflection of the image; fluorescence channels are
   spectral, NOT directional -> no channel swap), and the +-4 px translation must be applied identically to target and input (or document
   why a mismatch is harmless) -- check how prep() applies shift and handle it correctly.
3. Tests (CPU, tiny): (a) flag off -> identical forward outputs and loss vs the unpatched file for a fixed seed/batch; (b) flag on -> one
   forward/backward step finite, decoder output shape, grads reach the encoder; (c) packer on 64 rows reproduces a hand computation;
   (d) target/input stay registered under shift and diag-aug (unit test with a synthetic asymmetric image).
Deliverables in this folder: train_lf.py (full patched copy), bsccm/dense_aux.py, pack_fluor_aux.py, pack_fluor_aux.sbatch, tests/ +
their logs, README.md (exact commands; what you verified; open risks), and a unified diff vs the current repo file (train_lf.diff).
Do not submit any SLURM job. Do not edit anything outside this folder. Keep the final message <= 400 words.
