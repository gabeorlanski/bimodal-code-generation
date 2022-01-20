#!/bin/bash
#SBATCH --job-name=train-adv-code
#SBATCH --output=./logs/train.log
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "

source /ext3/env.sh
conda activate adversarial-code

torchrun --nproc_per_node=4 train.py $name $task --config greene_config -force -hydra model='$model' objective='$objective' +disable_cache=True
"