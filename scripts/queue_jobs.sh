#!/bin/bash
# Queue Slurm Jobs

pre_jid=$(sbatch --parsable --job-name=negcodeparrot_pre train.sbatch NegativeSOCodeParrot so pretrain_config lvwerra/codeparrot lm  "++training.num_train_epochs=1 ++task.dump_name='negative' ++training.dataloader_num_workers=16 ++num_proc=16")
echo "Submitted PreTrain (id=$pretrain_jid)"
train_jid=$(sbatch --parsable --dependency=afterok:$pre_jid --job-name=negcodeparrot1 \
  train.sbatch NegativeSOCodeParrot mbpp greene_config lvwerra/codeparrot lm \
  "is_checkpoint=True +model_path=best_models/SO.NegativeSOCodeParrot")
echo "Submitted Train (id=$train_jid)"
eval_jid=$(sbatch --parsable --job-name=negcodeparroteval --dependency=afterok:$train_jid eval.sbatch best_models/MBPP.NegativeSOCodeParrot/ validation,test 25)
echo "Submitted Eval $eval_jid to run after $1(id=$train_jid)"

