#!/bin/bash
# Queue Single Slurm

train_jid=$(sbatch --parsable --job-name=$1 train.sbatch $3 $4 $5 $6 ${@:7})
echo "Submitted Train $train_jid"
eval_jid=$(sbatch --parsable --job-name=$1 --dependency=afterok:$train_jid eval.sbatch best_models/$2.$3/ validation,test 25)
echo "Submitted Eval $eval_jid to run after $train_jid"