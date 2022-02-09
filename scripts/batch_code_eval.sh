#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform.ParrotSmall.HighQual.FineTune-preds:v2 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform.ParrotSmall.Exceptions.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform.ParrotSmall.General.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform.ParrotSmall.Negative.FineTune-preds:v0 16
