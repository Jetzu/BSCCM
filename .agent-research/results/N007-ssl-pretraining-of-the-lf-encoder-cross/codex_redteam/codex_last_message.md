**Verdict: FAIL.** Findings and minimal fixes are in [REDTEAM.md](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N007-ssl-pretraining-of-the-lf-encoder-cross/codex_redteam/REDTEAM.md).

- **Critical:** collapsed embeddings receive perfect retrieval scores.
- **Major:** the duplicate mask ignores pairs with different marker sets.
- **Major:** supervised resume can report old metrics under a new protocol.
- **Major:** target calibration consumes held-out labels.
- **Major:** screen SigLIP labels repeated positive pairs as negatives.

No GPU or full training/export pipeline was run. Scratch was deleted; source files were unchanged.