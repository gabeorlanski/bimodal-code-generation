#!/bin/bash
# Queue Slurm Jobs

parse_jid=$(sbatch --parsable parse_so.sbatch)
pre_jid=$(sbatch --parsable --dependency=afterok:$parse_jid --job-name=negcodeparrot \
  train.sbatch NegativeSOCodeParrot so lvwerra/codeparrot lm pretrain_config \
  training.max_steps=10000 training.log_steps=50 training.save_steps=250 training.eval_steps=250)
echo "Submitted PreTrain (id=$pretrain_jid)"
train_jid=$(sbatch --parsable --dependency=afterok:$pre_jid --job-name=negcodeparrot1 \
  train.sbatch NegativeSOCodeParrot mbpp lvwerra/codeparrot lm \
  is_checkpoint=True +model_path=best_models/SO.NegativeSOCodeParrot)
echo "Submitted Train (id=$train_jid)"
eval_jid=$(sbatch --parsable --job-name=negcodeparrot2 --dependency=afterok:$train_jid eval.sbatch best_models/MBPP.NegativeSOCodeParrot/ validation,test 25)
echo "Submitted Eval $eval_jid to run after $1(id=$train_jid)"
