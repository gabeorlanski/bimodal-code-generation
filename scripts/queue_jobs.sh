#!/bin/bash
# Queue Slurm Jobs

sbatch --job-name=codeparrot train.sbatch CodeParrot mbpp \
  lvwerra/codeparrot lm
sbatch --job-name=gptneo train.sbatch GPTNeo mbpp EleutherAI/gpt-neo-1.3B lm
sbatch --job-name=gpt2 train.sbatch GPT2 mbpp gpt2-xl lm
sbatch --job-name=codeparrot_small single_train.sbatch CodeParrotSmall mbpp \
  lvwerra/codeparrot-small lm training.batch_size=4
