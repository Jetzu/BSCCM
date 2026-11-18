#!/bin/bash
# Submit the exact full grid in dependency order. Run on the GPU cluster.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
R=/hnvme/workspace/v181bb10-bsccm/runs
RIDGE="$R/n054_reference/ridge.npz"
submit() {
    local dependency="$1"; shift
    local job
    if [[ -n "$dependency" ]]; then
        job=$(sbatch --parsable --chdir="$HERE" --export="ALL,N054_PACKAGE=$HERE" --dependency="afterok:$dependency" "$HERE/n054_gpu.sbatch" "$@")
    else
        job=$(sbatch --parsable --chdir="$HERE" --export="ALL,N054_PACKAGE=$HERE" "$HERE/n054_gpu.sbatch" "$@")
    fi
    job="${job%%;*}"
    echo "Submitted $job: $*" >&2
    echo "$job"
}
REF=$(submit '' lf_spaces --arm A1 --holdout batch1 --sigma-bridge 0 --w-sig 0 --out "$R/n054_A1_s0_w0_batch1" --ridge-cache "$RIDGE")
declare -a RUNS=() VALS=() ARMS=() HOLDS=() SIGMAS=() WEIGHTS=()
for holdout in batch1 batch0; do
    for spec in A1:0:0 A1:0.1:0 A2:0:0 A2:0.1:0 A3:0:0 A3:0:0.5; do
        IFS=: read -r arm sigma weight <<< "$spec"
        out="$R/n054_${arm}_s${sigma}_w${weight}_${holdout}"
        flags=(--arm "$arm" --holdout "$holdout" --sigma-bridge "$sigma" --w-sig "$weight" --out "$out" --ridge-cache "$RIDGE")
        if [[ "$arm:$sigma:$weight:$holdout" == A1:0:0:batch1 ]]; then lf="$REF"; else lf=$(submit "$REF" lf_spaces "${flags[@]}"); fi
        gs=$(submit "$lf" gs_encoder "${flags[@]}")
        flow=$(submit "$gs" flow "${flags[@]}")
        val=$(submit "$flow" eval --validation-only "${flags[@]}")
        RUNS+=("$out"); VALS+=("$val"); ARMS+=("$arm"); HOLDS+=("$holdout"); SIGMAS+=("$sigma"); WEIGHTS+=("$weight")
    done
done
VAL_DEP=$(IFS=:; echo "${VALS[*]}")
SELECT=$(submit "$VAL_DEP" select --runs "${RUNS[@]}" --out "$R/n054_selection.json")
for i in "${!RUNS[@]}"; do
    arm="${ARMS[$i]}"; holdout="${HOLDS[$i]}"; out="${RUNS[$i]}"
    flags=(--arm "$arm" --holdout "$holdout" --sigma-bridge "${SIGMAS[$i]}" --w-sig "${WEIGHTS[$i]}" --out "$out" --ridge-cache "$RIDGE")
    ev=$(submit "$SELECT" eval "${flags[@]}")
    if [[ "$arm:${WEIGHTS[$i]}:$holdout" == A3:0:batch1 ]]; then
        submit "$ev" completion "${flags[@]}" >/dev/null
    fi
    submit "$ev" clinical --arm "$arm" --holdout "$holdout" --run "$out" --out "${out}/clinical" >/dev/null
done
for holdout in batch1 batch0; do
    out="$R/n054_A0_${holdout}"
    ev=$(submit "$SELECT" eval --arm A0 --holdout "$holdout" --out "$out" --ridge-cache "$RIDGE")
    submit "$ev" clinical --arm A0 --holdout "$holdout" --run "$out" --out "${out}/clinical" >/dev/null
done
echo "Selection job: $SELECT. Choices freeze in $R/n054_selection.json before any held-out-slide jobs."
