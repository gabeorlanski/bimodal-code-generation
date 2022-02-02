#!/bin/bash
# Queue Slurm Jobs

parse_job=$(sbatch --parsable parse_so.sbatch)
pre_jid=$(sbatch --parsable --dependency=afterok:$parse_job --job-name=negcodeparrot_pre train.sbatch NegativeSOCodeParrot so pretrain_config lvwerra/codeparrot lm  "++training.max_steps=10000 ++training.logging_steps=25 ++training.save_steps=500 ++training.eval_steps=500 ++training.num_train_epochs=1 ++task.dump_name='negative'")
echo "Submitted PreTrain (id=$pretrain_jid)"
train_jid=$(sbatch --parsable --dependency=afterok:$pre_jid --job-name=negcodeparrot1 \
  train.sbatch NegativeSOCodeParrot mbpp greene_config lvwerra/codeparrot lm \
  "is_checkpoint=True +model_path=best_models/SO.NegativeSOCodeParrot")
echo "Submitted Train (id=$train_jid)"
eval_jid=$(sbatch --parsable --job-name=negcodeparrot2 --dependency=afterok:$train_jid eval.sbatch best_models/MBPP.NegativeSOCodeParrot/ validation,test 25)
echo "Submitted Eval $eval_jid to run after $1(id=$train_jid)"

