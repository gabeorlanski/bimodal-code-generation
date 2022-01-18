#!/bin/bash
# Simple Experiment

#torchrun --nproc_per_node=$1 train.py $2 $3 -force -hydra model=$4 objective=$5 ${@:7}
python evaluate.py outputs/$3/train/$2 validation -seqs 1000 \
  -hydra +generation.num_return_sequences=100 +device=$6