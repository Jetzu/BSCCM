# Codex WRITER brief v3: N054 second-round fixes (Claude reproduced each P1 first-hand)

v3 starts as a copy of your v2 files in THIS folder (codex/v3/); edit them here; leave v2 untouched. Same rules (repo + data read-only,
scratch /tmp/codex_n054/, CPU smoke only). Re-review scripts: /scratch/claude-215432/-home-hpc-v181bb-v181bb10/2a8ea17d-13e8-41ba-8d05-26ceffe3390d/scratchpad/n054rt2/ (d2, d6, d7).
Production v2 LF-unit output exists at /hnvme/workspace/v181bb10-bsccm/runs/n054v2_lf (read-only reference; v3 uses prefix n054v3).

P1-A in-sample map training (reproduced, d6.py): E_G is fitted on gs_train (split 0) and the A1-GS / direct maps are trained on the
  SAME cells' z_G; on a legacy model of the same recipe cos(z_G, z_L) is .717 on its training cells vs .558 on held-out FOVs of the same
  slide (R@1 .54 vs .20). Required: primary A1-GS and direct maps are trained on OUT-OF-SAMPLE z_G = source-slide split-1 GS cells
  (E_G never saw them), fixed sigma 0, fixed small step budget (e.g. 600 steps, report the curve on a 20 % FOV-grouped hold-out of
  split 1 for information only, no selection on it), initialised from the LF-pair pretrained map. Keep the in-sample split-0 versions as
  secondary rows (labelled). Parity: the direct map must ALSO be pretrained on LF pairs (direct regression P2(f) -> P1(f) on LF cells,
  same capacity), then fine-tuned identically; A1-GS starts from A1-LF. Report both pre- and post-fine-tune numbers.
P1-B frame-dependent OT in A2 (reproduced, d7.py): the Hungarian cost compares coordinates of two independently initialised spaces;
  after a random rotation of one frame the true pairing is recovered 0.000 and matched-cell autofl r is -.03. Required: A2 primary =
  minibatch entropic Gromov-Wasserstein coupling (rotation-invariant; intra-space cosine-distance matrices of the source batch z_G and
  the target batch P1(f) of different non-GS LF cells; e.g. POT ot.gromov.entropic_gromov_wasserstein if importable in the container
  (check: apptainer is not available to you, so implement a small torch entropic-GW solver yourself, Peyre et al. 2016, with a unit test
  that recovers a random rotation+permutation of the same point set), then sample pairs from the coupling for flow matching. Keep the
  Euclidean-cost A2 as a secondary row labelled frame-dependent. Add a sanity row: the GW coupling recovers same-cell identity when source
  = P2(f) and target = P1(f) of the SAME LF cells (report identity rate vs chance).
P1-C completion controls (reproduced, d2.py: a linear stack of LF-only and labels-only predictions matches or beats fused in about half
  the rows). Required: add (e) stacked control = linear regression on [pred_LF, pred_labels_MLP] fitted on SOURCE split-1 rows (never
  test), (f) plain MLP on [f, y*mask, mask] with no shared space and no marker encoder (same hidden width/depth as the fused head, same
  masking regime). hidden_information_supported = fused > max(a, c_mlp, c_linear, d, e, f). Report per-row gains vs each control.
P2: run python with `-I -B` everywhere (no __pycache__ in the notebook folder); disclose in report.md that LF-side training includes
  the held-out draw's non-GS unstained cells (~2 % of rows: 6,604 slide-19 / 5,088 slide-2) -- keep as is (shared LF unit), but write
  per-holdout counts into the audit JSON; default prefix n054v3.
Keep everything else from v2. Update README (exact sbatch commands), report.md (per finding: change + evidence), GPU estimate, and
re-run the full CPU smoke incl. the SIGTERM resume test and the GW unit test. Handback <= 1.2k tokens.
