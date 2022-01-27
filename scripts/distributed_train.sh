#!/bin/bash
python scripts/prepare_train_environment.py $2 $3 --config greene_config -force \
  -hydra model='$4' objective='$5' +disable_cache=True \
  ${@:6}
torchrun --nproc_per_node=$1 train.py $2 $3 --config greene_config -force \
  -hydra model='$4' objective='$5' +disable_cache=True \
  ${@:6}