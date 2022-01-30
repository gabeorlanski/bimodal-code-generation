#!/bin/bash

bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.GPTNeo1300M 16;
bash scripts/docker_eval_code.sh nyu-code-research/adversarial-code MBPP.CodeParrot1500M 16;
