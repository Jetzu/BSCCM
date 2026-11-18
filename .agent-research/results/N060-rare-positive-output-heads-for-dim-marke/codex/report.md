Implemented N060–N062 in `train_lf.py` plus `bsccm_heads.py`; repository untouched. `PATCH.diff` installs both. Exact commands and implementation choices are in README.md and run_arms.sh.

Recommend these six fixed arms, each seeds 0/1 × b1to0/b0to1, with the stated N031-initialized r50 recipe and frozen v10 targets:

| Item | Flags |
|---|---|
| N060 | `--head hlgauss --hl-mse .1` |
| N060 | `--head twopart --pos-q .98 --gate-w .5` |
| N061 | `--fov-re .01 --re-shrink 1 --w-autofl 0` |
| N061 | `--pair-loss .5 --fov-batch 64 --w-autofl 0` |
| N061 | `--w-autofl 0` |
| N062 | `--panel-head --re-shrink 1` |

Budget: 24 runs, approximately **7.2 GPU-h** at .4 h b1to0 / .2 h b0to1, before modest head/sampling overhead. Each arm costs 1.2 GPU-h. Add 1.2 h if fresh two-seed controls are needed; reuse only recipe-matched v10 controls. For a winning pair arm, a sampler-matched `--fov-batch 64 --pair-loss 0 --w-autofl 0` control separates loss from sampling. Keep the N059 single-model/ensemble +.0054 screening bar; inspect CD123 per seed and direction before attributing a mean gain.

Evaluation function is byte-identical. All heads return 13 label-unit predictions; training adapters never enter forward prediction. New defaults are omitted from `_cfg()`. Source calibration, thresholds, bin edges and adapter identities are pinned in configuration/checkpoint buffers. Active new recipes reject held-out-image adaptation options and require one source draw. Panel and categorical heads are alternatives; HL-Gauss plus FOV affine adapters is explicitly rejected.

CPU evidence: all six arms × both directions ran a real ResNet-18 with `--max-train 64`, using 528 compacted real cells, ordinary normalization, training and evaluation. Mathematical checks cover masking, Gaussian bin integration, gate balance, pair cancellation/duplicate exclusion, adapter identity/source restrictions and panel gradient sharing. Finished checkpoints resume. Default model tensors, training loss and saved P exactly match a CPU-placement adaptation of the original; an original-format checkpoint resumes with new defaults. Tiny-count metric summaries are NaN under unchanged minimum-count rules; losses and all prediction entries are finite. No GPU execution or quality claim.

Corrections/caveats from the data and N059 review:

- Labels are detrended, asinh-transformed and normalized assay coordinates, not molecular abundance. Pooled values are merged spectral readouts; adding normalized singles is unjustified. The panel arm therefore mixes latent factors and learns separate monotone assay decoders. Factor identification remains weak; pooled supervision may hurt.
- Source draw 1 has **zero** pooled CD123/HLA-DR/CD14 labels and no cls3/cls10 gates. Its cls3 strata are predictions from an unsupervised class head, so interpret the mixed-effects arm cautiously. The missing pooled target stays masked, with placeholder histogram support.
- Quantile gates identify the top 2% of observed training intensities, not biological positivity. Balanced BCE does not calibrate prevalence. HL support clipping can lose extreme-tail amplitude, especially CD123; the requested expectation MSE cannot exceed the support.
- `--w-autofl 0` drops the entire auxiliary target, including unstained source cells. It does not selectively remove only co-stained labels.
- FOV-centering diagnostics remove biology as well as nuisance. Their .04 gap is not guaranteed recoverable. Existing normalization constants retain the harness's historical label-free cross-slide preprocessing; this patch introduces no held-out-draw fit.

HL-Gauss attribution: the original [Imani & White paper is 2018](https://proceedings.mlr.press/v80/imani18a.html); the [expanded investigation is Imani et al., 2024](https://arxiv.org/abs/2402.13425), alongside [Farebrother et al., 2024](https://proceedings.mlr.press/v235/farebrother24a.html). The nuisance tables are a regression adaptation of [Rodrigues & Pereira's crowd layer](https://ojs.aaai.org/index.php/AAAI/article/view/11506), not a demonstrated BSCCM correction.
