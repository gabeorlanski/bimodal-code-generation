#!/bin/bash
# Queue Slurm Jobs

sbatch --job-name=codeparrot train.sbatch CodeParrot1500M mbpp \
  lvwerra/codeparrot lm
sbatch --job-name=gptneo train.sbatch GPTNeo1300M mbpp EleutherAI/gpt-neo-1.3B lm
sbatch --job-name=gpt2 train.sbatch GPT2_1588M mbpp gpt2-xl lm
