#!/bin/bash
# Queue Slurm Jobs
rm -rf sbatch_logs
mkdir "sbatch_logs"
sbatch --job-name='codeparrot_exceptions_execute' eval_code.sbatch eval_results/MBPP MBPP.CodeParrotSmall.Exceptions.FineTune
echo ""
sbatch --job-name='codeparrot_general_execute' \
  eval_code.sbatch eval_results/MBPP MBPP.CodeParrotSmall.General.FineTune
echo ""
sbatch --job-name='codeparrot_highqual_execute'  \
  eval_code.sbatch eval_results/MBPP MBPP.CodeParrotSmall.HighQual.FineTune
echo ""
