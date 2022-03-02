#!/bin/bash
#SBATCH --output=./sbatch_logs/%x_parse_dump.out
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --account=cds
#SBATCH -c 16

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "

source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)
python scripts/parse_so_data.py parse $1 16
"