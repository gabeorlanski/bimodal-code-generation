#!/bin/bash
echo $(pwd)
python scripts/prepare_train_environment.py $2 $3 --config $4 -force --override-str $7 \
  -hydra model=$5 objective=$6 +disable_cache=True \
  ${@:8}
torchrun --nproc_per_node=$1 train.py $2 $3 --config $4 -force --override-str $7 \
  -hydra model=$5 objective=$6 +disable_cache=True \
  ${@:8}