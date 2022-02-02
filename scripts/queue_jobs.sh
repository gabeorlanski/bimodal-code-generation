#!/bin/bash
# Queue Slurm Jobs
pre_jid=$(sbatch --parsable --job-name=negcodeparrot_pre train.sbatch NegativeSOCodeParrot so lvwerra/codeparrot lm pretrain_config ++training.max_steps=10000 ++training.log_steps=50 ++training.save_steps=250 ++training.eval_steps=250 ++training.num_train_epochs=1 ++task.dump_name='negative')
echo "Submitted PreTrain (id=$pretrain_jid)"
train_jid=$(sbatch --parsable --dependency=afterok:$pre_jid --job-name=negcodeparrot1 \
  train.sbatch NegativeSOCodeParrot mbpp lvwerra/codeparrot lm greene_config\
  is_checkpoint=True +model_path=best_models/SO.NegativeSOCodeParrot)
echo "Submitted Train (id=$train_jid)"
eval_jid=$(sbatch --parsable --job-name=negcodeparrot2 --dependency=afterok:$train_jid eval.sbatch best_models/MBPP.NegativeSOCodeParrot/ validation,test 25)
echo "Submitted Eval $eval_jid to run after $1(id=$train_jid)"

