#!/bin/bash
# Evalue un checkpoint en planification sur une grille solveur x contexte
# (appele par x-cluster/cluster.py, depuis ~/psc/code, env conda active, PSC_RUN_DIR fixe).
#
#   bash scripts/cluster/run_eval_grid.sh <env> <job_d_entrainement|random|oracle> [overrides hydra...]
#
# Grille : solveur in {gd, cem} x contexte H in {1, 2, 3} (variables SOLVERS, CONTEXTS).
# Resultats : $PSC_RUN_DIR/eval_<solveur>_h<H>.json. Meme seed d'evaluation pour tout :
# memes 100 problemes depart/but pour tous les modeles et toutes les variantes.
# random et oracle ne dependent ni du solveur ni du contexte : une seule evaluation.
set -u
ENV=$1; TRAIN_JID=$2; shift 2
SOLVERS=${SOLVERS:-gd cem}
CONTEXTS=${CONTEXTS:-1 2 3}
case "$TRAIN_JID" in
  random|oracle) W=$TRAIN_JID; SOLVERS=cem; CONTEXTS=1 ;;
  *)
    W=$(ls -t /tmp/$USER/psc/$TRAIN_JID/swm_home/checkpoints/$TRAIN_JID/weights_epoch_*.pt 2>/dev/null | head -1)
    [ -z "$W" ] && W=$(ls -t $HOME/psc/runs/$TRAIN_JID/weights_epoch_*.pt 2>/dev/null | head -1)
    if [ -z "$W" ]; then echo "checkpoint introuvable pour $TRAIN_JID sur $(hostname)"; exit 3; fi ;;
esac
echo "checkpoint : $(hostname):$W" | tee "$PSC_RUN_DIR/checkpoint.txt"
export STABLEWM_HOME=$HOME/psc/swm_home   # datasets (lecture seule ici)

cd scripts/plan
STOP=0; PID=; FAILS=0
trap '[ -n "$PID" ] && kill -TERM $PID 2>/dev/null; STOP=1' TERM INT
for S in $SOLVERS; do
  for H in $CONTEXTS; do
    TAG=${S}_h${H}
    [ "$W" = random ] || [ "$W" = oracle ] && TAG=$W
    for TRY in 1 2; do   # 2 essais : erreur CUDA sporadique observee dans le CEM
        [ $STOP -eq 1 ] && exit 143
        python eval_wm.py --config-name "$ENV" policy="$W" solver="$S" context="$H" video=false \
            output.json="$PSC_RUN_DIR/eval_$TAG.json" hydra.run.dir="$PSC_RUN_DIR/hydra_$TAG" "$@" &
        PID=$!
        wait $PID; RC=$?
        while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done
        PID=
        [ $RC -eq 0 ] && break
        echo "=== $TAG : echec (rc=$RC), essai $TRY"
    done
    [ $RC -eq 0 ] || FAILS=$((FAILS + 1))
  done
done
exit $FAILS
