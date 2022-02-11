#!/bin/bash
# Queue Slurm Jobs


sbatch --job-name='Uniform.HypeParam.ParrotSmall_HighQual.LongSteps_execute' eval_code.sbatch eval_results/MBPP MBPP.Uniform.HypeParam.ParrotSmall.HighQual.LongSteps.FineTune


sbatch --job-name='Uniform.HypeParam.ParrotSmall_HighQual.HighLR_execute' eval_code.sbatch eval_results/MBPP MBPP.Uniform.HypeParam.ParrotSmall.HighQual.HighLR.FineTune