#!/usr/bin/env bash
# Login-node cron (helma2): BSCCM offline wandb runs -> wandb.ai/Protein-Design-Tools/BSCCM. Compute nodes have no internet.
# Syncs every offline run; a run whose .wandb file is untouched for >60 min (finished) and already synced is deleted afterwards
# to keep /hnvme inodes low (metrics also live in runs/*/result.json and logs/).
set -uo pipefail
export HOME=/home/hpc/v181bb/v181bb10
export PATH=$HOME/miniforge3/envs/triad/bin:/usr/local/bin:/usr/bin:/bin
export WANDB_SILENT=true
D=/hnvme/workspace/v181bb10-bsccm/wandb/wandb
LOG=$HOME/BSCCM/logs/wandb_sync_cron.log
exec 9>/tmp/bsccm_wandb_sync.lock; flock -n 9 || exit 0
cd /tmp || exit 1
for r in "$D"/offline-run-*; do
  [ -d "$r" ] || continue
  f=$(ls "$r"/run-*.wandb 2>/dev/null | head -1); [ -n "$f" ] || continue
  if [ -e "$f.synced" ] && [ -z "$(find "$f" -mmin -60)" ]; then rm -r -- "$r"; echo "$(date '+%F %T') pruned $(basename "$r")" >>"$LOG"; continue; fi
  timeout 600 wandb sync --project BSCCM "$r" >>"$LOG" 2>&1; echo "$(date '+%F %T') sync $(basename "$r") exit=$?" >>"$LOG"
done
tail -n 2000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
