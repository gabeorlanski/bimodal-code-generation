#!/bin/bash
# Queue Slurm Jobs
sbatch --parsable --job-name=CodeParrotSmall_Exceptions_eval \
	--dependency=afterok:$train_jid3 eval.sbatch \
	best_models/MBPP.CodeParrotSmall.Exceptions.FineTune validation,test 25

sbatch --parsable --job-name=CodeParrotSmall_General_eval \
	--dependency=afterok:$train_jid4 eval.sbatch \
	best_models/MBPP.CodeParrotSmall.General.FineTune validation,test 25
sbatch --parsable --job-name=CodeParrotSmall_HighQual_eval \
	--dependency=afterok:$train_jid5 eval.sbatch \
	best_models/MBPP.CodeParrotSmall.HighQual.FineTune validation,test 25