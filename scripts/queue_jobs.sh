#!/bin/bash
# Queue Slurm Jobs

parse_jid=$(sbatch parse_so.sbatch)
sbatch --parsable --dependency=afterok:parse_jid --job-name=negcodeparrot \
  train.sbatch NegativeSOCodeParrot so lvwerra/codeparrot lm pretrain_config \
  training.max_steps=11750 training.log_steps=50 training.save_steps=250 training.eval_steps=250
