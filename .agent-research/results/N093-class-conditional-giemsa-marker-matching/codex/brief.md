# Codex FIX ROUND brief (N093). Hat: fix (writer). Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex ONLY; repo read-only; scratch /tmp/n093_codex2. CPU only.
Round-1 result (job 1467846, /hnvme/workspace/v181bb10-bsccm/runs/n093_class_match_s0/) + Claude red-team (/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/redteam_claude.md):
the PBC-anchored GS classifier routes 96.9-99.2 % of BSCCM held-out Giemsa cells to class 1 (granulocyte) -> F1/F2 agreement with the LF
teacher ~0, autofl r .01-.07, MLL23 .516 (a domain collapse PBC -> BSCCM Giemsa, cf. N008), and F2's rank coordinate separates PBC from
BSCCM images (AUROC 1.00 = domain, not biology). The user's idea therefore was not tested fairly. Add, keeping F1/F2 unchanged:
F1-oracle (BSCCM metrics only; an UPPER BOUND, label it so): route each held-out paired GS cell by the class of ITS OWN LF image through the
same LF teacher's cls3 head; marker prediction = that class's mean profile (and the soft version with the teacher's class probabilities).
F1-bsccm: a GS cell-type classifier on frozen DinoBloom-B features trained on BSCCM Giemsa cells of the TRAINING slide only, with targets
= the LF teacher's cls3 class of their paired LF image (soft labels allowed), plus the PBC half-A anchors as in the harness (state the mix);
evaluate exactly like F1 on all metrics (MLL23_15, primary16, PBC10, all25, autofl r on the held-out slide (use N058's sanity metric
definition too), per-marker agreement with the LF teacher). Report the class fractions of every router on every evaluation set next to
the gated/LF-teacher reference fractions. Same teachers t0-t2, both holdouts, seed 0, deterministic.
Deliver: patched modules, updated tests (CPU fixtures), README commands (48-core CPU job), <= 250-word handback.
