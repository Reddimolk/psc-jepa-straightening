#!/bin/bash
# Lance un entrainement sur une machine des salles info (appele par x-cluster/cluster.py,
# depuis ~/psc/code, env conda active, PSC_RUN_DIR = ~/psc/runs/<job>).
#
#   bash scripts/cluster/run_job.sh keep|local [--eval <env>[,override,...]] <overrides hydra...>
#
# Quota NFS de 30 Go : les checkpoints (Lightning + stable-pretraining + poids par epoque)
# vont sur le disque local /tmp de la machine, pas dans le home. Le dataset reste lu depuis
# le cache NFS (lien symbolique). Seuls metrics.jsonl / log / config restent dans PSC_RUN_DIR ;
# avec `keep`, les derniers poids (~70 Mo) y sont aussi copies en fin de job.
# Avec --eval <env>, le checkpoint final est evalue en planification juste apres
# l'entrainement, sur la meme machine (resultat : $PSC_RUN_DIR/eval.json). Overrides de
# l'evaluation separes par des virgules : --eval reacher,eval.num_eval=100
set -u
KEEP=$1; shift
EVAL=
if [ "${1:-}" = --eval ]; then EVAL=$2; shift 2; fi
EVAL_ENV=${EVAL%%,*}
EVAL_ARGS=$( [ "$EVAL" != "$EVAL_ENV" ] && echo "${EVAL#*,}" | tr ',' ' ' )
JID=$(basename "$PSC_RUN_DIR")
LOCAL=/tmp/$USER/psc/$JID
mkdir -p "$LOCAL/swm_home" "$LOCAL/spt"
ln -sfn "$HOME/psc/swm_home/datasets" "$LOCAL/swm_home/datasets"
export STABLEWM_HOME=$LOCAL/swm_home SPT_CACHE_DIR=$LOCAL/spt
echo "$LOCAL" > "$PSC_RUN_DIR/local_dir.txt"

# SIGTERM (timeout de cluster.py) -> transmis au processus en cours
PID=
trap '[ -n "$PID" ] && kill -TERM $PID 2>/dev/null; STOP=1' TERM INT
STOP=0
run() {
    "$@" &
    PID=$!
    wait $PID; RC=$?
    while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
    PID=
    return $RC
}

run python scripts/train/lewm.py output_model_name="$JID" subdir="$JID" \
    hydra.run.dir="$PSC_RUN_DIR/hydra" "$@"
RC=$?

W=$(ls -t "$STABLEWM_HOME/checkpoints/$JID"/weights_epoch_*.pt 2>/dev/null | head -1)
if [ -n "$W" ]; then
    echo "$(hostname):$W" > "$PSC_RUN_DIR/weights_location.txt"
    if [ "$KEEP" = keep ]; then
        cp "$W" "$STABLEWM_HOME/checkpoints/$JID/config.json" "$PSC_RUN_DIR/" 2>/dev/null
    fi
fi

if [ -n "$EVAL" ] && [ $RC -eq 0 ] && [ $STOP -eq 0 ] && [ -n "$W" ]; then
    # 2 essais : erreur CUDA sporadique observee dans le CEM (smoke test du 29/09)
    for TRY in 1 2; do
        echo "=== evaluation en planification ($EVAL_ENV $EVAL_ARGS) de $W, essai $TRY"
        ( export STABLEWM_HOME=$HOME/psc/swm_home
          cd scripts/plan && exec python eval_wm.py --config-name "$EVAL_ENV" policy="$W" video=false \
              output.json="$PSC_RUN_DIR/eval.json" hydra.run.dir="$PSC_RUN_DIR/hydra_eval" $EVAL_ARGS ) &
        PID=$!
        wait $PID; ERC=$?
        while kill -0 $PID 2>/dev/null; do wait $PID; ERC=$?; done
        echo "eval exit $ERC (essai $TRY)" > "$PSC_RUN_DIR/eval_exit_code"
        if [ $ERC -eq 0 ] || [ $STOP -eq 1 ]; then break; fi
    done
fi
exit $RC
