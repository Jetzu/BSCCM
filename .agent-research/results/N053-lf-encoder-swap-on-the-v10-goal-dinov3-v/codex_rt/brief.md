# Codex RED-TEAM brief: N053 encoder adapter (Claude wrote it; you attack it). Write only into this folder; scratch only /tmp/codex_n053rt/.

Repo /home/hpc/v181bb/v181bb10/BSCCM is READ-ONLY for you. Review:
- bsccm/encoders.py (new): LFBackbone = learned per-LED gate -> bilinear resize 128 -> 256/224 -> timm ViT with RGB patch embedding
  adapted to 24 input channels (DINOv3 S/B via timm pretrained_cfg_overlay file=local safetensors; DinoBloom-B from a DINOv2 teacher
  checkpoint; Cytoland VSCyto2D ConvNeXtV2-tiny via bsccm/cytoland.py).
- bsccm/cytoland.py (new): VSCyto2D encoder as timm convnextv2_tiny with a 2x2 stride-2 stem, zero frozen dw-conv biases, LN eps 1e-5,
  1-channel stem repeated to 24 channels / 24. Converted weights: /hnvme/workspace/v181bb10-bsccm/models/cytoland/converted/.
- scripts/train_lf.py changes (diff in train_lf.diff here): --arch dinov3_s|dinov3_b|dinobloom_b|cytoland, --enc-lr-mult (separate
  param group with lower lr and a two-element OneCycleLR max_lr), --drop-path, led_weights in result.json; legacy single-group
  optimizer kept when enc-lr-mult == 1 so old checkpoints resume.
Context: this is the GOAL harness (batch-held-out LF -> marker Pearson r; protocols b1to0 / b0to1; labels targets_v5cx_v10.npz).
Running jobs use: --epochs 10 --bs 256 --lr 5e-4 --enc-lr-mult 0.1 --w-cls 0.5 --include-gs 0.
Attack: (1) are the pretrained weights really loaded (timm overlay path, in_chans adaptation for Eva/DINOv3 with RoPE + register
tokens, DinoBloom key mapping, pos-embed size at 224) or silently random? (2) does anything leak held-out-draw information or change
the evaluation vs the r50 baseline (same rows, same eval, same labels)? (3) optimizer/scheduler bugs (param groups, OneCycle with
lists, weight decay on gate/norm/pos-embed, frozen params in the optimizer), resume/cfg consistency, channels_last on ViT, autocast;
(4) Cytoland conversion correctness (eps, stem, dw bias), (5) anything making the comparison unfair to the ViTs (lr, resolution,
normalisation: our input is log-normalised per-LED, not ImageNet-normalised) that a reviewer would flag. You have no GPU and no
apptainer; you may build a CPU venv in /tmp/codex_n053rt with torch CPU + timm==1.0.31 + safetensors to load the real weights and
check numerically. Report findings ranked by severity with file:line, a concrete failure scenario and the evidence you ran.
Handback <= 1.5k tokens in report.md.
