#!/bin/bash
# Queue Slurm Jobs
rm -rf sbatch_logs
mkdir "sbatch_logs"
sbatch --job-name='codeparrot_exceptions_execute' \
  eval_code.sbatch eval_outputs/MBPP MBPP.CodeParrotSmall.Exceptions.Finetune
echo ""
sbatch --job-name='codeparrot_general_execute' \
  eval_code.sbatch eval_outputs/MBPP MBPP.CodeParrotSmall.General.FineTune
echo ""

sbatch --job-name='codeparrot_highqual_execute'  \
  eval_code.sbatch eval_outputs/MBPP MBPP.CodeParrotSmall.HighQual.FineTune
echo ""

sbatch --job-name='codeparrot_negative_execute' \
  eval_code.sbatch eval_outputs/MBPP MBPP.CodeParrotSmall.Negative.FineTune
echo ""