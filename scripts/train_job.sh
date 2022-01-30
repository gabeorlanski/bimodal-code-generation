#!/bin/bash
# Queue Single Slurm

jid $(sbatch --job-name=$1 train.sbatch $3 $4 $5 $6 ${@:7})
sbatch --job-name=$1 --dependency=afterok:$jid eval.sbatch best_models/$2.$3/ validation,test 25