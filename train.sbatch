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
python scripts/prepare_train_environment.py $1 $2 --config $3 -force --override-str $6 \
  -hydra model='$4' objective='$5' +disable_cache=True \
  ${@:7}
torchrun --nproc_per_node=4 train.py $1 $2 --config $3 -force --override-str $6 \
  -hydra model='$4' objective='$5' +disable_cache=True \
  ${@:7}
"