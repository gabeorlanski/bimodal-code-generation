#!/bin/bash
#SBATCH --output=./sbatch_logs/%x_train.out
#SBATCH --account=cds
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mem=148G
#SBATCH -c 16
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
python scripts/prepare_train_environment.py $1 -force
/home/go533/.local/bin/deepspeed --num_gpus=4 train.py from_config $1
"
