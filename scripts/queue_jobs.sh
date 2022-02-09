#!/bin/bash
# Queue Slurm Jobs
train_jid0=$(sbatch --parsable --job-name=Uniform32Epoch.ParrotSmall_Negative_finetune train_single_gpu.sbatch generated_experiments/MBPP.Uniform32Epoch.ParrotSmall.Negative.FineTune.yaml)
echo "Submitted Train (id=$train_jid0)"
eval_jid0=$(sbatch --parsable --job-name=Uniform32Epoch.ParrotSmall_Negative_eval \
	--dependency=afterok:$train_jid0 eval.sbatch \
	best_models/MBPP.Uniform32Epoch.ParrotSmall.Negative.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid0 to run after (id=$train_jid0)"
echo ""
