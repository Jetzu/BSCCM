# Fact-check of readme/README.md (Claude red-team, 2026-11-24)

Scope: README.md (796 lines, written 18:26) against INDEX.md, INDEX_ARCHIVE.md, KNOWLEDGE.md, GOAL.md, RULES.md, IDEAS.md, items N001-N088,
figures/overview/CAPTIONS.md (incl. the two late edits), figures/overview/data/*.csv, results/N0xx-*/ (jobs.txt, scorers, csv).
Nothing in the repo was changed.

What was checked:
- About 170 numeric values: every value in the summary (L3-10), "What counts as progress" (L65-92), "What worked for each task" (L94-113),
  "Reproduce" (L115-243), plus 2-6 values for every timeline item N001-N088.
- First-hand recomputation (read-only, from runs/*/eval_preds.npz and targets_v5cx_v10.npz): N053 .7001/.6723/.6862, MM3 .7734;
  N053 with Cytoland seed 1 .7043/.6685/.6864; N084 .7030/.6793/.6911, MM3 .7764; control .6730. The table at L176-180 is correct.
- All 12 training and SSL commands at L212-238 match the job ledgers byte for byte (N048 1375401-04, N053 1376834/35 and 1377120/21,
  N078 1430134, N083 1439071/1439073, N084 1440201). The reconstructed N031 SSL command (L203) matches the archived offline wandb config
  of n031_ssl_aug_ex0 (job 1295538: arm aug, resnet50, 40 ep, bs 512, lr .002, lam .05, proj 256, rows all, exclude_batch [0]).
- All relative links resolve locally. Every script, module and workspace path named at L250-335 exists.
- The N069-N072 "record unavailable" entries are correct: no item, result folder or index row exists.
- The N065 .6929 is correctly called a planned descriptive secondary (L563), which matches late edit 2.

Most numbers are correct. The errors below are about attribution, wording and verdicts, plus one claim the late CAPTIONS edit rules out.

## Errors

| # | README line | README text | Correct value / text | Source |
|---|---|---|---|---|
| 1 | 102 (table "What worked", LF↔Giemsa, "Best-supported result") | "Paired embeddings recovered exact cells within a slide at R@10 around .380–.400." | This must not be presented as cell-level recovery or as a best-supported result. Within-slide R@10 .38–.40 may come partly from a field-of-view shortcut: paired images share background and focus, and there are only about 3 paired cells per field. The audit is pending, and the cross-slide number (.030) is the trustworthy one. | figures/overview/CAPTIONS.md:240, :255; results/N088-*/synthesis.md:9 |
| 2 | 681 (Figure 8 text) | "Paired retrieval succeeds more often within a slide." | Same caveat is missing: "possibly helped by a field-of-view shortcut, check pending". | CAPTIONS.md:255 |
| 3 | 108 | "On the N084 marker breakdown, CD16 reaches about .900. CD19 (about .350), HLA-DR (.470), CD56 (.520) and CD123 (.560)" | N084's own values (mean of the two directions) are CD16 .900, CD19 .360, HLA-DR .480, CD56 .531, **CD123 .596**. The quoted .35/.47/.52/.56 are approximate means over four models (the ResNet-50 and ConvNeXt-S seed means, N053, N084), not the N084 breakdown. Recomputed first-hand. | figures/overview/data/per_target.csv:69-89 (Ensemble N084 rows); CAPTIONS.md:87 ("four models") |
| 4 | 44 | "Only some pooled cells of draw 0 have gates" | Gates exist only for the draw-0 pooled ("all") cells, and almost all of them have one: about 10,137-10,161 of 10,210 cells. No other cell has a gate. "Only some" wrongly suggests a small subset. | KNOWLEDGE.md:43 (K002, 10,161); IDEAS.md:6 (10,210 'all' cells in batch 0); items/N004:21 |
| 5 | 386 | "retrieved the exact LF cell among ten candidates only .011 of the time" | This is R@10 over all 5,697 cells of the held-out slide (chance R@1 .00018, so chance R@10 is about .0018). It is not a 10-way choice. An ML reader would take "among ten candidates" to mean chance .1 and read .011 as below chance. | items/N009:13; README's own definition at L90 |
| 6 | 626-627 | "Two feature-export jobs were logged ... No final decoder comparison is recorded." | This is out of date: the source now lists 14 jobs. They are 2 feature exports, the decoder sweep (1442606), 6 image-AE runs (1442607-12), the marker-prior sandbox (1442613), the flow decoder (1442504) and 3 equal-label-count controls (1442617-19). The item and job ledger were updated at 18:28, after the README (18:26). N088 synthesis.md:18 already records "flow decoder (done: no co-expression gain)". | items/N087:58-63; results/N087-*/jobs.txt:1-14; results/N088-*/synthesis.md:18 |
| 7 | 483 | "Training against a teacher's predictions for missing entries raised single-model r from .643 to .6529." | The student also used physics augmentation (phys aug .7). Augmentation alone gives +.0025, so the pseudo-labels add about +.008 of the +.0103. | items/N040:39-44 |
| 8 | 380-381 | N007: "... The stronger SSL recipe went forward to fine-tuning." | The recorded verdict is **worse** (goal_effect worse, value .479). SSL linear probes did not beat the supervised ResNet-18 (batch-held-out .479 vs .483; FOV .522 vs .707). Cross-illumination views were worse than augmentation-only, which refutes the item's hypothesis. The README implies a positive outcome. | items/N007:4, :14-16; KNOWLEDGE.md:56 (K017) |
| 9 | 106 | "At fixed predictions, v10 and local-background labels agreed at only about .320 for CD19." | This is agreement between two label versions on held-out cells; no predictions are involved. CD19 is .24/.40 per direction (mean .32). "At fixed predictions" suggests a prediction-label correlation. | CAPTIONS.md:98-99; items/N064:27 |
| 10 | 536-537 | N057: "The trend stayed within the specified seed floor, motivating a matched-strength test of the distillation target." | N057 and N058 ran at the same time in two sessions, and both followed from N056. N057 did not motivate N058 ("Complements N057"; N057: "decided by N058"). | items/N058:9; items/N057:37-38 |
| 11 | 566 | "On two pooled-panel slides of the latter test draw, CD45 scored .881 and .578." | This is the pooled target **grp:CD45**, not the single CD45 target (.778 for N084). | KNOWLEDGE.md:10-11 (K043) |
| 12 | 436-437 | N025 is described only through the Giemsa chain. | The item's registered GOAL verdict is missing: **FLAT on the GOAL** (v6-trained SSL r50 .639 vs v5 .6404). | items/N025:29, :33 |
| 13 | 596 | "[N077] · Review external datasets ..." | N077 is an analysis (a data scout), not one of the five project reviews. Calling it "Review" in a history that numbers Reviews 1-5 (N038, N047, N059, N066, N079) is misleading. | INDEX.md:55 (type analysis) |

## Commands and reproduction

No command differs from its jobs.txt. The following are problems with how the commands can be used:
- **L211-238, archived run directories.** The rebuild commands write to the historical `--out` directories (for example runs/n048_v9_base_b10),
  which already hold finished ckpt.pt and eval_preds.npz files. The trainers are preempt-safe and resume from `<out>/ckpt.pt`
  (scripts/gpu_job.sbatch:2), so running the commands as printed would resume or overwrite the archived record runs rather than produce an
  independent rebuild. The only warning is in prose (L196-197); the code blocks carry none.
- **L200 and L340-341, dependencies outside the repo.** gpu_job.sbatch runs `~/.claude/skills/work-log/bin/preempt-run.sh`, which is not
  in the repo, and hard-codes `cd $HOME/BSCCM` and `/home/hpc/v181bb/v181bb10/BSCCM/logs` (scripts/gpu_job.sbatch:9,11,14). The README calls it
  "the work-log preemption helper" and does not say that a new team member cannot run it from the repo.
- **L138 and L145, Python environment.** "the cluster's CPU Python environment" is `$HOME/miniforge3/envs/triad`, a personal conda environment
  of another project. Any Python with NumPy and pandas works.
- **L154 and L203, shell behaviour.** `exit 1` closes the reader's interactive shell. `$W` in the step-2 block is defined only in the first
  block.
- **L254, container base image.** The Apptainer recipe builds from `/hnvme/workspace/v181bb10-salad/torch29_cu128.sif` (env/bsccm.def:2),
  which is in another project's workspace.
- **L346, L630, L694-707, L795-796, links that will break on GitHub.** The links to items N087 and N088 and to results/N088-*/context.md
  point at untracked files (`git status` shows `??`), so they will 404 on GitHub until those files are committed.

## Overclaims and wording an ML reader would misread

1. **L102 and L387, within-slide retrieval.** Within-slide retrieval (.38-.40, and N009's .349) is reported without the field-of-view
   shortcut caveat (late edit 1). N009's within-slide screen has the same problem.
2. **L10, Giemsa student claim in the summary.** "teaching the student the label-free representation gave better clinical class rankings"
   is stated without caveats. The teacher-paired test gives p = .07 (ns), and the result passes only the registered effect-size rule
   (.113 > .074) with 3 teachers. It is also measured on MLL23, which the project itself calls a reused development benchmark
   (KNOWLEDGE.md:80, items/N025:33). L103 qualifies the claim; the summary does not.
3. **L9, L101 and L606, co-expression.** None of the ImageStream regime differences is significant at 3 donors (CAPTIONS.md:133, :137), and
   the README never says "ns". It also omits that over all 4 within-population pairs, single-marker training has the lowest pair error
   (.308 vs .389 for full labels; items/N086:28-29, :40-43; fig4_multimarker.json notes). Only CD4-CD8 exclusivity and the registered
   36-pair metric favour full labels.
4. **L524-525, N053 record.** "This replicated ensemble gain set the official record" hides that Cytoland was chosen post hoc among 4
   candidate third members on the test draws (items/N053:52; items/N086:51). The ".6864 replication" is a fresh seed of the member that was
   already chosen, not an independent confirmation.
5. **L105 and L466, different-cell bags.** "Bags of different cells yielded correlations around zero" holds only for the signed mean. The
   check3 correction reports mean |r| .05-.15 and single markers up to |r| .48 of both signs, from systematic field-of-view components rather
   than chance (items/N034:109-111).
6. **L104 and L495, residual flow.** The new-draw residual-flow r .594 comes from the development direction, the same one used in N034, with
   no frozen confirmation. The item failed its pre-registered success rule (items/N044:14-20, :51-56). "recovered" reads stronger than that.
7. **L108, per-marker values.** The values described as N084's are cross-model approximations; see error 3.
8. **L703-704 and L710, current work.** This section is out of date compared with results/N088-*/synthesis.md (written 18:31). The
   conditional flow decoder is already done (no co-expression gain), and N090 (gentle fine-tuning) has been launched; neither is reflected.

## PRIVATE DATA: none in the README text

- No e-mail addresses: the only "@" characters are bash `"${py[@]}"`.
- No tokens, keys or secrets.
- No supervisor or mentor names; L711 uses "the supervising lab", as the brief requires.
- Note 1: the README links to items/N004 (L25, L39, L45, L51, L707, L715), and that item names the supervisor and two mentors
  (items/N004:9). This is repo content rather than README text, but it becomes visible through the link once the repo is shared.
- Note 2: the cluster account ID "v181bb10" appears in every workspace path (L143, L212-238, L264). It is not personal data in the brief's
  sense.
