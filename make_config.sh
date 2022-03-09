#!/bin/bash
# Generate experiments
singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
if [ $2 -eq "0" ]
  then
    python scripts/create_experiments.py ${1:-experiment_card} conf -overwrite
  else
    python scripts/create_experiments.py ${1:-experiment_card} conf
fi

"