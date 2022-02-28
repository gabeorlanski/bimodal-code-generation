#!/bin/bash
#SBATCH --account=cds
#SBATCH --output=./sbatch_logs/%x_tensorsize.log
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH -c 48
#SBATCH -p cs
echo "Starting Tensorize"
singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)
python tensorize.py $1 $2 $3 $4 48 --config $5
"