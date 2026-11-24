# Codex FIX-ROUND brief (N081). Hat: fix round -- root-cause + build in this folder (/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/codex_fix) ONLY; repo read-only; scratch /tmp/n081_fix; CPU only.
Start from YOUR delivered /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/codex/train_lf.py (+ tests) and fix the findings of the Claude red-team /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N081-two-stem-joint-source-training-bsccm-led/agents_redteam.md (read it fully;
reproduction scripts in /tmp/n081_rt if present). Required fixes:
1 BLOCKER: per-source BatchNorm RUNNING statistics (domain-specific BN): keep two buffer sets for every BN layer of the shared trunk/stem
  (running_mean, running_var, num_batches_tracked); BSCCM forwards use/update only the BSCCM set, IFC forwards only the IFC set (swap
  buffers around the IFC forward in training and around evaluate_ifc); BN affine weights stay shared. evaluate() must see exactly the
  BSCCM buffers that BSCCM-only training would have produced from the same BSCCM batches. Save both sets in the checkpoint; strict resume.
  Test: with --ext-ifc, after N steps the BSCCM buffers equal those of a run whose IFC forward is skipped but RNG-identical.
2 MAJOR: donors = run-name prefix (regex ^sample\d+): sample1_early + sample1_late = donor sample1. --ext-test-donor/--ext-train-donors
  take DONOR names; exclusion by donor; assert no test-donor run in training. Test with the real naming.
3 MAJOR: IFC affine init per shared marker from training statistics: scale = sd_ifc_train / sd_bsccm_train-prediction-scale proxy --
  concretely scale0 = sd(IFC train labels) / sd(BSCCM training-row v10 labels of the mapped target), offset0 = mean_ifc - scale0 * mean_bsccm;
  exclude the affine (and BN/bias params if the harness already does so -- check) from weight decay.
4 minors: separate torch.Generator for IFC init/augmentation/shuffle so the BSCCM RNG stream equals the matched control; IFC stem in the
  encoder param group under --enc-lr-mult; reject --adabn with --ext-ifc; vectorise the IFC D4 augmentation (group by transform);
  log BSCCM and IFC gradient norms separately (keep one clip but report how often it activates).
5 Packer statistics: add --stat-donors (default sample0,sample1) to /home/hpc/v181bb/v181bb10/BSCCM/scripts/pack_sbiad452.py COPY in this
  folder (pack_sbiad452.py) so cofactors and chan_scale are fitted on those donors only; record in pack.json.
Keep flag-off byte-equality (rerun the N074 suite + your tests against the live repo file sha256 4cf89b09...). Deliver: train_lf.py,
train_lf.diff (vs the live repo file), pack_sbiad452.py (+diff), tests/ + logs, README.md, <= 300-word handback.
NOTE (Claude, first-hand check of MAJOR-2): v10 label SDs on labelled rows are CD14 .98/1.00, CD19 .36/.52, CD3 1.00/1.01, CD45 .98/1.00,
CD56 .92/.84 (draw 1 / draw 0) -- the red-team's .26/.20/.49/.92/.61 did NOT reproduce. Keep fix 3 (data-driven init computed from the
actual masked training rows of the training draw, so it is right either way) and print the init values.
