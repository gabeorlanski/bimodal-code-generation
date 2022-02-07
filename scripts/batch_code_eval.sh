#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.Exceptions.FineTune 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.General.FineTune 16
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall.HighQual.FineTune 16
