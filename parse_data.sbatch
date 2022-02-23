#!/bin/bash
#SBATCH --account=cds
#SBATCH --output=./sbatch_logs/job_parse_so.log
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH -c 32
#SBATCH -p cs
echo "Starting Parse"
singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
echo $(pwd)
python scripts/parse_so_data.py -out data/parsed_so parse data/stack_exchange/stackoverflow 32 raw_data \
	--cleaner NONE --do-not-filter
"