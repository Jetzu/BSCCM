# N091 DINO encoder

Apply [train_ssl_lf.diff](train_ssl_lf.diff) to `scripts/train_ssl_lf.py` and place [the regression suite](tests/test_train_ssl_lf.py) in `tests/`. The [full patched file](train_ssl_lf.py) is byte-identical to the applied diff.

## Training commands after integration

Run each command inside its one-H100 allocation. Register the job ID in N091 and the node’s offline W&B sync lineage at submission.

```bash
cd /home/hpc/v181bb/v181bb10/BSCCM
work=/hnvme/workspace/v181bb10-bsccm
apptainer exec --nv -B /hnvme "$work/env/bsccm.sif" python scripts/train_ssl_lf.py \
  --objective dino --arm aug --arch resnet50 --rows all --exclude-batch 0 \
  --epochs 40 --bs 512 --views 2 --local-crops 0 --seed 0 --out "$work/runs/n091_dino_r50_ex0"
apptainer exec --nv -B /hnvme "$work/env/bsccm.sif" python scripts/train_ssl_lf.py \
  --objective dino --arm aug --arch resnet50 --rows all --exclude-batch 1 \
  --epochs 40 --bs 512 --views 2 --local-crops 0 --seed 0 --out "$work/runs/n091_dino_r50_ex1"
```

| Excluded draw | Training cells | Expected wall time on one H100 |
|---|---:|---:|
| 0 | 274,980 | ~2 h 26 min |
| 1 | 137,961 | ~47 min |

These estimates multiply the matched 40-epoch ResNet-50 SIGReg times by 1.3 for the teacher forwards. Combined estimate is 3.2 H100-hours. [N031 timings](../../../notebook/items/N031-inductive-ssl-control-re-pretrain-the-ss.md) · [N088 P3 cost assumption](../../N088-literature-review-user-11-24-label-free/lit_A_labelfree_representations.md#p3-rank-3-clustering-objective-in-domain-ssl-dino-style-self-distillation-with-prototypes-no-local-crops-vs-sigreg)

The teacher tracks its own global-view batch-normalization statistics. In epoch one, prototype directions learn while their magnitudes stay at one. Optional local crops alternate between globals with up to eight pixels of centre jitter per axis. Epoch logs include mean per-target entropy and KL to uniform, plus KL of pooled prototype usage to uniform.

## CPU verification

Run the suite from this handoff directory or the repository root after integration.

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES='' apptainer exec -B /hnvme \
  /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python -B tests/test_train_ssl_lf.py
```

In a sandbox without FUSE, unpack this SIF’s SquashFS partition into fresh scratch and pass the resulting directory to the same command in place of the SIF.

```bash
mkdir -p /tmp/n091_codex
unsquashfs -processors 48 -no-progress -no-xattrs -offset 40960 \
  -d /tmp/n091_codex/rootfs /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif
```

Five tests passed using the unpacked SIF under Apptainer. The legacy, default and legacy-resume SIGReg trajectories matched exactly (7.510198, 8.083667, 13.620529, rounded). After SIGTERM, DINO’s next loss matched exactly (5.712509632110596). A 48-channel ResNet-50 completed three 128×128 CPU bf16 steps with 4,096 prototypes and passed the downstream strict loader. Fixtures checked both exclusions at image read time. Production-data training, H100 memory/timing and live W&B sync remain unmeasured.
