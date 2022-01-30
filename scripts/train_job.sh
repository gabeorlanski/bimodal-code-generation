#!/bin/bash
# Queue Single Slurm

jid1=$(sbatch --job-name=$1 train.sbatch $3 $4 $5 $6 ${@:7})
echo $jid1
sbatch --job-name=$1 --dependency=afterok:$jid1 eval.sbatch best_models/$2.$3/ validation,test 25