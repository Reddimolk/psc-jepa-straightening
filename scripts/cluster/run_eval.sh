#!/bin/bash
# Evalue un checkpoint en planification (taux de succes MPC, scripts/plan/eval_wm.py).
# Appele par x-cluster/cluster.py (depuis ~/psc/code, env conda active, PSC_RUN_DIR fixe).
#
#   bash scripts/cluster/run_eval.sh <pusht|reacher> <job_d_entrainement|random> [overrides hydra...]
#
# Le checkpoint est cherche d'abord sur le disque local de la machine
# (/tmp/$USER/psc/<job>/..., la ou run_job.sh l'a ecrit), puis dans ~/psc/runs/<job>/
# (copie NFS des jobs lances en `keep`). Resultat : $PSC_RUN_DIR/eval.json.
set -u
ENV=$1; TRAIN_JID=$2; shift 2
if [ "$TRAIN_JID" = random ]; then
    W=random   # reference : politique aleatoire (plancher du taux de succes)
else
W=$(ls -t /tmp/$USER/psc/$TRAIN_JID/swm_home/checkpoints/$TRAIN_JID/weights_epoch_*.pt 2>/dev/null | head -1)
[ -z "$W" ] && W=$(ls -t $HOME/psc/runs/$TRAIN_JID/weights_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$W" ]; then
    echo "checkpoint introuvable pour $TRAIN_JID sur $(hostname)"; exit 3
fi
fi
echo "checkpoint : $(hostname):$W" | tee "$PSC_RUN_DIR/checkpoint.txt"
export STABLEWM_HOME=$HOME/psc/swm_home   # datasets (lecture seule ici)

cd scripts/plan
python eval_wm.py --config-name "$ENV" policy="$W" video=false \
    output.json="$PSC_RUN_DIR/eval.json" hydra.run.dir="$PSC_RUN_DIR/hydra_eval" "$@" &
PID=$!
trap 'kill -TERM $PID 2>/dev/null' TERM INT
wait $PID; RC=$?
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
exit $RC
