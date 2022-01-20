#!/bin/bash
#SBATCH --job-name=train-adv-code
#SBATCH --output=./logs/eval.out
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "

source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)

python evaluate.py $1 $2 -seqs 30 -hydra +generation.num_return_sequences=$3 +training.batch_size=$4 +device=0 +generation.max_length=512
"