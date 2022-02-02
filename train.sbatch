#!/bin/bash
#SBATCH --output=./logs/%x_train.out
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx8000
#SBATCH --mem=180G
#SBATCH -c 4
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)

name=${1}
task=${2}
model=${3}
objective=${4}
config=${5}
hydra_args=${@:6}

python scripts/prepare_train_environment.py $name $task --config $config -force \
  -hydra model='$model' objective='$objective' +disable_cache=True $hydra_args

torchrun --nproc_per_node=1 train.py $name $task --config $config -force \
  -hydra model='$model' objective='$objective' +disable_cache=True $hydra_args
"