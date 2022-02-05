#!/bin/bash
#SBATCH --output=./logs/%x_train.out
#SBATCH --account=cds
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=rtx8000
#SBATCH --mem=180G
#SBATCH --account=cds
#SBATCH -c 48
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
python scripts/prepare_train_environment.py $1 -force
torchrun --nproc_per_node=4 --n train.py from_config $1
"
