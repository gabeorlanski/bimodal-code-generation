#!/bin/bash
#SBATCH --output=./sbatch_logs/%x.out
#SBATCH --time=24:00:00
#SBATCH --account=cds
#SBATCH --mem=128G
#SBATCH -c 32
#SBATCH --kill-on-invalid-dep=yes

singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)
echo "$1"
python code_eval.py $1 32
"