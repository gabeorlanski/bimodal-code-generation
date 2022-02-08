#!/bin/bash
# Queue Slurm Jobs
rm -rf sbatch_logs
eval_jid1=$(sbatch --parsable --job-name=codeparrot_exceptions_eval eval.sbatch \
	best_models/MBPP.CodeParrotSmall.Exceptions.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid1"
sbatch --job-name='codeparrot_exceptions_execute' --dependency=afterok:$eval_jid1 \
  eval_code.sbatch MBPP.CodeParrotSmall.Exceptions.Finetune
echo ""
eval_jid1=$(sbatch --parsable --job-name=codeparrot_general_eval eval.sbatch \
	best_models/MBPP.CodeParrotSmall.General.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid1"
sbatch --job-name='codeparrot_general_execute' --dependency=afterok:$eval_jid1  \
  eval_code.sbatch MBPP.CodeParrotSmall.General.FineTune
echo ""

eval_jid1=$(sbatch --parsable --job-name=codeparrot_highqual_eval eval.sbatch \
	best_models/MBPP.CodeParrotSmall.HighQual.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid1"
sbatch --job-name='codeparrot_highqual_execute' --dependency=afterok:$eval_jid1  \
  eval_code.sbatch MBPP.CodeParrotSmall.HighQual.FineTune
echo ""

eval_jid1=$(sbatch --parsable --job-name=codeparrot_negative_eval eval.sbatch \
	best_models/MBPP.CodeParrotSmall.Negative.FineTune validation,test 100 "remove_input_ids=True")
echo "Submitted Eval $eval_jid1"
sbatch --job-name='codeparrot_negative_execute' --dependency=afterok:$eval_jid1  \
  eval_code.sbatch MBPP.CodeParrotSmall.Negative.FineTune
echo ""