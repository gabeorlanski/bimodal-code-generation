#!/bin/bash
# Queue Slurm Jobs
#rm -rf sbatch_logs
#mkdir sbatch_logs
#
#for task in "MBPP" "SO"
#do
#  for ABLATION in "FullData" "SODump10KSteps" "Uniform32Epoch"
#  do
#    for SPLIT in "Exceptions" "HighQual" "Negative" "General"
#    do
#      if [ "$task" == "SO" ]; then
#        STEP="PreTrain"
#      else
#        STEP="FineTune"
#      fi
#      if [ "$ABLATION" == "SODump10KSteps" ]; then
#        full_name=$ABLATION
#      elif [ "$ABLATION" == "Uniform32Epoch" ] && [ "$task" == "SO" ]; then
#        full_name="Uniform.ParrotSmall"
#      else
#        full_name="$ABLATION.ParrotSmall"
#      fi
#
#      model_name="$task.$full_name.$SPLIT.$STEP"
#      echo "Model= $model_name"
#      eval_jid1=$(sbatch --parsable --job-name=${model_name}_eval eval.sbatch best_models/$model_name test human_eval 100 1024 "remove_input_ids=True")
#      echo "Submitted $eval_jid1"
#      sbatch --job-name=${model_name}_execute --dependency=afterok:$eval_jid1 eval_code.sbatch eval_results/HUMAN_EVAL $model_name
#      echo ""
#
#    done
#  done
#done

# Command for Base_CodeParrotSmall
sbatch --job-name='Base_CodeParrotSmall_execute' eval_code.sbatch eval_results/HUMAN_EVAL Baseline.CodeParrotSmall
echo ""


sbatch --job-name='Base_CodeParrotSmall.Replication_execute' eval_code.sbatch eval_results/HUMAN_EVAL Baseline.CodeParrotSmall.Replication
echo ""

sbatch --job-name='Base_CodeParrotSmall.HighTemp_execute' eval_code.sbatch eval_results/HUMAN_EVAL Baseline.CodeParrotSmall.HighTemp
echo ""