#!/bin/bash
# Queue Slurm Jobs
sbatch --parsable --job-name=CodeParrotSmall_Exceptions_eval eval.sbatch best_models/MBPP.CodeParrotSmall.Exceptions.FineTune validation,test 100

sbatch --parsable --job-name=CodeParrotSmall_General_eval  eval.sbatch best_models/MBPP.CodeParrotSmall.General.FineTune validation,test 100
sbatch --parsable --job-name=CodeParrotSmall_HighQual_eval eval.sbatch best_models/MBPP.CodeParrotSmall.HighQual.FineTune validation,test 100