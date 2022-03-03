#!/bin/bash
#SBATCH --account=cds
#SBATCH --output=./sbatch_logs/%x_consolidate.log
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -p cs
echo "Starting Tensorize"
singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)
python scripts/parse_so_data.py consolidate $1 $2 $3 -val ${4:-10000}
"