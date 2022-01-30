#!/bin/bash
# Queue Slurm Jobs

bash scripts/single_job.sh codeparrot MBPP CodeParrot mbpp lvwerra/codeparrot lm
bash scripts/single_job.sh gptneo MBPP GPTNeo mbpp EleutherAI/gpt-neo-1.3B lm
bash scripts/single_job.sh gpt2 MBPP GPT2 mbpp gpt2-xl lm
bash scripts/single_job.sh codeparrot_small MBPP CodeParrotSmall mbpp \
  lvwerra/codeparrot-small lm training.batch_size=4
