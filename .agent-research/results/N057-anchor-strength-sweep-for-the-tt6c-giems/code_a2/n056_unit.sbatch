#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --signal=B:USR1@900
#SBATCH -J n056v2
#SBATCH -o /hnvme/workspace/v181bb10-bsccm/runs/n056v2_%j.out
set -euo pipefail
: "${N056_UNIT:?Export N056_UNIT=train or eval}"
: "${N056_TEACHER:?Export N056_TEACHER=0, 1, or 2}"
N056_ARM=${N056_ARM:-H}
N056_SEED=${N056_SEED:-0}
N056_CODEDIR=${N056_CODEDIR:-/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N057-anchor-strength-sweep-for-the-tt6c-giems/code_a2}
N056_RUNROOT=${N056_RUNROOT:-/hnvme/workspace/v181bb10-bsccm/runs/n057_a2}
case "$N056_TEACHER" in 0|1|2) ;; *) exit 64;; esac
case "$N056_ARM" in T|H|H_matched) ;; *) exit 64;; esac
case "$N056_ARM:$N056_TEACHER:$N056_SEED" in
  T:0:0|T:0:1|T:0:2|T:1:0|T:1:1|T:2:0|T:2:1|H:[012]:[01]|H_matched:[012]:0) ;; *) exit 64;; esac
export N056_UNIT N056_TEACHER N056_ARM N056_SEED N056_CODEDIR N056_RUNROOT
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
trainout="$N056_RUNROOT/t${N056_TEACHER}/${N056_ARM}_s${N056_SEED}_train"
evalout="$N056_RUNROOT/t${N056_TEACHER}/${N056_ARM}_s${N056_SEED}"
mkdir -p "$N056_RUNROOT"
cd "$N056_CODEDIR"
case "$N056_UNIT" in
 train)
  [ "$N056_ARM" != T ] || exit 64
  args=("$N056_CODEDIR/n056_hub_student.py" --teacher "$N056_TEACHER" --seed "$N056_SEED" --arm "$N056_ARM" --out "$trainout") ;;
 eval)
  args=("$N056_CODEDIR/n056_eval.py" --stage all --teacher "$N056_TEACHER" --seed "$N056_SEED" --arm "$N056_ARM" --hub "$trainout" --out "$evalout") ;;
 *) exit 64 ;;
esac
# Environment parameters survive preempt-chain replay. Python returns 143 only
# after a transactional cursor or optimizer+RNG checkpoint has been saved.
exec "$HOME/.claude/skills/work-log/bin/preempt-run.sh" -- \
 apptainer exec --nv --bind /hnvme/workspace/v181bb10-bsccm \
 /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I -B "${args[@]}"
