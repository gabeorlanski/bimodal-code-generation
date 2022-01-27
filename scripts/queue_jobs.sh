#!/bin/bash
# Queue Slurm Jobs

sbatch --job-name=codeparrot train.sbatch CodeClippy1300M mbpp \
  lvwerra/codeparrot \
  lm \
  training.learning_rate=5e-5
