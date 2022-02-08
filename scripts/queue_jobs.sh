#!/bin/bash
# Queue Slurm Jobs
rm -rf sbatch_logs
mkdir "sbatch_logs"
eval_jid1=$(sbatch --parsable --job-name=codeparrotsmall_eval  eval.sbatch \
	best_models/MBPP.CodeParrotSmall validation,test 100)
echo "Submitted Eval $eval_jid1 to run "
sbatch --job-name='codeparrotsmall_execute' --dependency=afterok:$eval_jid1 eval_code.sbatch eval_results/MBPP MBPP.CodeParrotSmall
echo ""

eval_jid2=$(sbatch --parsable --job-name=codeparrot_eval  eval.sbatch \
	best_models/MBPP.CodeParrot validation,test 25)
echo "Submitted Eval $eval_jid2 to run "
sbatch --job-name='codeparrotsmall_execute' --dependency=afterok:$eval_jid2 eval_code.sbatch eval_results/MBPP MBPP.CodeParrot
echo ""
