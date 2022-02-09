#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform32Epoch.ParrotSmall.General.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform32Epoch.ParrotSmall.HighQual.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code \
  MBPP.Uniform32Epoch.ParrotSmall.Exceptions.FineTune-preds:v0 16
