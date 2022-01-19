#!/bin/bash
# Simple Experiment

torchrun --nproc_per_node=$1 train.py $2 $3 -force -hydra model=$4 objective=$5 ${@:8}
python evaluate.py outputs/$3/train/$2 validation -seqs 200 \
  -hydra +generation.num_return_sequences=25 +device=$6
python evaluate.py outputs/$3/train/$2 test -seqs 200 \
  -hydra +generation.num_return_sequences=25 +device=$6

make build-docker
docker run --rm --name code_eval go-adversarial-code:latest python code_eval.py validation outputs/$3/train/$2 $7
docker run --rm --name code_eval go-adversarial-code:latest python code_eval.py test outputs/$3/train/$2 $7