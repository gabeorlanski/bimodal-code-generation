#!/bin/bash
#SBATCH --output=./sbatch_logs/%x_train.out
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH --kill-on-invalid-dep=yes
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
python scripts/prepare_train_environment.py $1 -force
python train.py from_config $1
"
