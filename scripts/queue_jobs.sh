#!/bin/bash
# Queue Slurm Jobs


for task in "MBPP" "SO"
do
  for ABLATION in "FullData" "SODump10KSteps" "Uniform32Epoch"
  do
    for SPLIT in "Exceptions" "HighQual" "Negative" "General"
    do
      if [ "$task" == "SO" ]; then
        STEP="PreTrain"
      else
        STEP="FineTune"
      fi
      if [ "$ABLATION" == "SODump10KSteps" ]; then
        full_name=$ABLATION
      else
        full_name="$ABLATION.ParrotSmall"
      fi

      model_name="$task.$full_name.$SPLIT.$STEP"
      echo "Model= $model_name"
      eval_jid1=$(sbatch --parsable --job-name=${model_name}_eval eval.sbatch best_models/$model_name test human_eval 100 1024 "remove_input_ids=True")
      echo "Submitted $eval_jid1"
      sbatch --job-name='${model_name}_execute' --dependency=afterok:$eval_jid1 eval_code.sbatch eval_results/HUMAN_EVAL $model_name
      echo ""

    done
  done
done