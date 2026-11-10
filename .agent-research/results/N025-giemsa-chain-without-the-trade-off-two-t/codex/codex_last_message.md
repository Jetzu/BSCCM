Verdict: headline numbers reproduce, but “trade-off removed” and “GS truth is eval-only” are overstated.

[Review report](./redteam_N025.md) contains seven numbered findings, reproduction commands, evidence, and concrete fixes.

The main findings are:

- Nonlinear preprocessing uses GS autofluorescence to construct non-GS training labels.
- TT6 clinical parity remains unproven: mean difference versus v4 is −.0111, with a three-seed paired 95% interval of [−.0393, +.0171].
- MLL23 has been reused to guide model choices.
- The aux interface has a latent schema-indexing bug; current files are aligned.

Within-slide Pearson, evaluation rows, headline metrics, and direct GS exclusions in the ridge/probe paths passed verification.

The report gives the integration plan. Neural retraining, GPU inference, and raw-image provenance were not checked. The live tree was untouched; scratch was deleted.