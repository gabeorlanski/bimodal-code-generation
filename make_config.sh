#!/bin/bash
# Generate experiments
rm -f generated_experiments/experiments.sh
singularity exec --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate adversarial-code
if [ $# -gt 1 ];
  then
    echo "Overwriting"
    python scripts/create_experiments.py ${1:-experiment_card} conf -overwrite
  else
    python scripts/create_experiments.py ${1:-experiment_card} conf
fi
"