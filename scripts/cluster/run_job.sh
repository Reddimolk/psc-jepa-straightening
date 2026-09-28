#!/bin/bash
# Lance un entrainement sur une machine des salles info (appele par x-cluster/cluster.py,
# depuis ~/psc/code, env conda active, PSC_RUN_DIR = ~/psc/runs/<job>).
#
#   bash scripts/cluster/run_job.sh keep|local <overrides hydra...>
#
# Quota NFS de 30 Go : les checkpoints (Lightning + stable-pretraining + poids par epoque)
# vont sur le disque local /tmp de la machine, pas dans le home. Le dataset reste lu depuis
# le cache NFS (lien symbolique). Seuls metrics.csv / log / config restent dans PSC_RUN_DIR ;
# avec `keep`, les derniers poids (~70 Mo) y sont aussi copies en fin de job.
set -u
KEEP=$1; shift
JID=$(basename "$PSC_RUN_DIR")
LOCAL=/tmp/$USER/psc/$JID
mkdir -p "$LOCAL/swm_home" "$LOCAL/spt"
ln -sfn "$HOME/psc/swm_home/datasets" "$LOCAL/swm_home/datasets"
export STABLEWM_HOME=$LOCAL/swm_home SPT_CACHE_DIR=$LOCAL/spt
echo "$LOCAL" > "$PSC_RUN_DIR/local_dir.txt"

python scripts/train/lewm.py output_model_name="$JID" subdir="$JID" \
    hydra.run.dir="$PSC_RUN_DIR/hydra" "$@" &
PID=$!
# SIGTERM (timeout de cluster.py) -> transmis a python, puis on range quand meme
trap 'kill -TERM $PID 2>/dev/null' TERM INT
wait $PID; RC=$?
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done

W=$(ls -t "$STABLEWM_HOME/checkpoints/$JID"/weights_epoch_*.pt 2>/dev/null | head -1)
if [ -n "$W" ]; then
    echo "$(hostname):$W" > "$PSC_RUN_DIR/weights_location.txt"
    if [ "$KEEP" = keep ]; then
        cp "$W" "$STABLEWM_HOME/checkpoints/$JID/config.json" "$PSC_RUN_DIR/" 2>/dev/null
    fi
fi
exit $RC
