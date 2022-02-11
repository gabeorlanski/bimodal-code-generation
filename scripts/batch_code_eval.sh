#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.SODump10KSteps.32.ParrotSmall.General.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.SODump10KSteps.32.ParrotSmall.Negative.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.SODump10KSteps.32.ParrotSmall.HighQual.FineTune-preds:v0 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.SODump10KSteps.32.ParrotSmall.Exceptions.FineTune-preds:v0 16
