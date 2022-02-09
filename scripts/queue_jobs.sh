#!/bin/bash
# Queue Slurm Jobs
# Command for FullData.ParrotSmall_Negative
train_jid0=$(sbatch --parsable --job-name=FullData.ParrotSmall_Negative_finetune train_single_gpu.sbatch generated_experiments/MBPP.FullData.ParrotSmall.Negative.FineTune.yaml)
echo "Submitted Train (id=$train_jid0)"
eval_jid0=$(sbatch --parsable --job-name=FullData.ParrotSmall_Negative_eval \
	--dependency=afterok:$train_jid0 eval.sbatch \
	best_models/MBPP.FullData.ParrotSmall.Negative.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid0 to run after $1(id=$train_jid0)"
sbatch --job-name='FullData.ParrotSmall_Negative_execute' --dependency=afterok:$eval_jid0 eval_code.sbatch eval_results/MBPP MBPP.FullData.ParrotSmall.Negative.FineTune
echo ""

# Command for FullData.ParrotSmall_HighQual
train_jid1=$(sbatch --parsable --job-name=FullData.ParrotSmall_HighQual_finetune train_single_gpu.sbatch generated_experiments/MBPP.FullData.ParrotSmall.HighQual.FineTune.yaml)
echo "Submitted Train (id=$train_jid1)"
eval_jid1=$(sbatch --parsable --job-name=FullData.ParrotSmall_HighQual_eval \
	--dependency=afterok:$train_jid1 eval.sbatch \
	best_models/MBPP.FullData.ParrotSmall.HighQual.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid1 to run after $1(id=$train_jid1)"
sbatch --job-name='FullData.ParrotSmall_HighQual_execute' --dependency=afterok:$eval_jid1 eval_code.sbatch eval_results/MBPP MBPP.FullData.ParrotSmall.HighQual.FineTune
echo ""
