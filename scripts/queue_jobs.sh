#!/bin/bash
# Queue Slurm Jobs

sbatch --job-name='FullData.ParrotSmall_Exceptions_execute' eval_code.sbatch eval_results/MBPP MBPP.FullData.ParrotSmall.Exceptions.FineTune

sbatch --job-name='FullData.ParrotSmall_General_execute' eval_code.sbatch eval_results/MBPP MBPP.FullData.ParrotSmall.General.FineTune

sbatch --job-name='FullData.ParrotSmall_HighQual_execute' eval_code.sbatch eval_results/MBPP MBPP.FullData.ParrotSmall.HighQual.FineTune

sbatch --job-name='FullData.ParrotSmall_Exceptions_execute' eval_code.sbatch eval_results/MBPP MBPP.FullData.ParrotSmall.Exceptions.FineTune

sbatch --job-name='SODump10KSteps.32.ParrotSmall_General_execute'eval_code.sbatch eval_results/MBPP MBPP.SODump10KSteps.32.ParrotSmall.General.FineTune

sbatch --job-name='SODump10KSteps.32.ParrotSmall_HighQual_execute' eval_code.sbatch eval_results/MBPP MBPP.SODump10KSteps.32.ParrotSmall.HighQual.FineTune
sbatch --job-name='SODump10KSteps.32.ParrotSmall_Exceptions_execute' eval_code.sbatch eval_results/MBPP MBPP.SODump10KSteps.32.ParrotSmall.Exceptions.FineTune
sbatch --job-name='SODump10KSteps.32.ParrotSmall_Negative_execute' eval_code.sbatch eval_results/MBPP MBPP.SODump10KSteps.32.ParrotSmall.Negative.FineTune