#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --signal=B:USR1@900
#SBATCH -J n056
#SBATCH -o /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/n056_%j.out
set -euo pipefail
: "${N056_UNIT:?Export N056_UNIT=train or eval}"
: "${N056_TEACHER:?Export N056_TEACHER=0, 1, or 2}"
N056_ARM=${N056_ARM:-H}
N056_CODEDIR=${N056_CODEDIR:-/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex}
N056_RUNROOT=${N056_RUNROOT:-$N056_CODEDIR/gpu_runs}
case "$N056_TEACHER" in 0|1|2) ;; *) exit 64;; esac
case "$N056_ARM" in T|H) ;; *) exit 64;; esac
export N056_UNIT N056_TEACHER N056_ARM N056_CODEDIR N056_RUNROOT
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$N056_RUNROOT/t${N056_TEACHER}/H_train"
cd "$N056_CODEDIR"
case "$N056_UNIT" in
 train)
  args=("$N056_CODEDIR/n056_hub_student.py" --teacher "$N056_TEACHER" --out "$N056_RUNROOT/t${N056_TEACHER}/H_train") ;;
 eval)
  args=("$N056_CODEDIR/n056_eval.py" --stage all --teacher "$N056_TEACHER" --arm "$N056_ARM" --hub "$N056_RUNROOT/t${N056_TEACHER}/H_train" --out "$N056_RUNROOT/t${N056_TEACHER}/$N056_ARM") ;;
 *) exit 64 ;;
esac
# Wrapper checkpoints/resubmits on TERM/USR1; Python saves optimizer+RNG or
# inference batch cursor, and exits 143. All parameters survive environment replay.
exec "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
 apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm \
 /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B "${args[@]}"
