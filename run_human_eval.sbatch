#!/bin/bash
#SBATCH --output=sbatch_logs/%x.out
#SBATCH --time=24:00:00
#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --qos=cds
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)
python human_eval.py $1 lm -n 24
"