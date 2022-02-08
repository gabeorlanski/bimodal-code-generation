#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.Negative.FineTune:v1 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.HighQual.FineTune:v1 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.General.FineTune:v1 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.Exceptions.FineTune:v1 16
