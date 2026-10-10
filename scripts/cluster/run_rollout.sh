#!/bin/bash
# Erreur de rollout a K pas d'un checkpoint (scripts/plan/rollout_error.py), sur la machine
# qui le detient. Appele par x-cluster/cluster.py (depuis ~/psc/code, env conda active).
#
#   bash scripts/cluster/run_rollout.sh <env> <job_d_entrainement> [options de rollout_error.py]
#
# Resultat : $PSC_RUN_DIR/rollout.json
set -u
ENV=$1; TRAIN_JID=$2; shift 2
W=$(ls -t /tmp/$USER/psc/$TRAIN_JID/swm_home/checkpoints/$TRAIN_JID/weights_epoch_*.pt 2>/dev/null | head -1)
[ -z "$W" ] && W=$(ls -t $HOME/psc/runs/$TRAIN_JID/weights_epoch_*.pt 2>/dev/null | head -1)
if [ -z "$W" ]; then echo "checkpoint introuvable pour $TRAIN_JID sur $(hostname)"; exit 3; fi
echo "checkpoint : $(hostname):$W" | tee "$PSC_RUN_DIR/checkpoint.txt"
export STABLEWM_HOME=$HOME/psc/swm_home
cd scripts/plan
exec python rollout_error.py --ckpt "$W" --data "$ENV" --out "$PSC_RUN_DIR/rollout.json" "$@"
