#!/bin/bash
# Queue Slurm Jobs

sbatch --job-name=codeparrot train.sbatch CodeParrot1500M mbpp lvwerra/codeparrot lm
sbatch --job-name=gptneo train.sbatch GPTNeo1300M mbpp EleutherAI/gpt-neo-1.3B lm
sbatch --job-name=gpt2 train.sbatch GPTNeo mbpp EleutherAI/gpt-neo-1.3B lm
