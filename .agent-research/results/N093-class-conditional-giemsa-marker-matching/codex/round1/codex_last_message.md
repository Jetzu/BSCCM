**F1 and F2 are implemented. Eight CPU fixture tests and a capped real-data t0 run passed.** The full comparison with T and H_matched remains unmeasured. [Fixture log](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex/tests/fixture.log:11)

| PBC half-A anchor | BSCCM LF class | cls10 gates |
|---|---|---|
| Lymphocyte | 0 · lymphocyte | 0 |
| Neutrophil | 1 · granulocyte | 1, 2, 3 |
| Monocyte | 2 · monocyte | 7, 8, 9 |

All three registered LF encoders have a trained `cls3` head. F1 mixes their class-mean marker profiles using GS class probabilities. F2 routes each image to one class and computes its marker profile by entropic OT. The matching cost uses within-class ranks of distances to the three lineage centroids, giving GS embeddings and marker vectors comparable coordinates. [Student implementation](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex/n093_match.py:73)

The real smoke produced all 25 clinical contrasts, same-cell autofluorescence r and twelve marker-agreement coefficients per student. It verified the full N058 evaluation manifests before scoring the capped subsets. Apptainer ran the unpacked contents of the requested SIF because this host lacks `/dev/fuse`. [Smoke audit](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex/tests/real_smoke/audit.json:34)

**Integration plan.** Run the documented 48-core CPU command for t0–t2 in both slide directions. The holdouts cover 5,697 and 10,015 distinct paired cells. Full t1/t2 teacher replay, reverse-slide performance and production runtime await verification. The deliverables are confined to N093 `codex/`. Notebook closure and commit remain with the integration reviewer. [Exact commands](/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex/README.md:64)