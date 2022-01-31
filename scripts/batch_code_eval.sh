#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.GPTNeo 16;
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrot 16;
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.GPT2 16;
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrotSmall 16;
