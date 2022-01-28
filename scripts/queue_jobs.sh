#!/bin/bash
# Queue Slurm Jobs

sbatch --job-name=codeparrot train.sbatch CodeParrot1500M mbpp \
  lvwerra/codeparrot lm \
  training.gradient_accumulation_steps=4
sbatch --job-name=gpt2 train.sbatch GPT2_1588M mbpp gpt2-xl lm \
  training.gradient_accumulation_steps=4
