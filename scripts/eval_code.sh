#!/bin/bash
# Simple Experiment

python download_artifact.py $1 artifacts

for split_file in artifacts/*.jsonl ;
do
  echo $split_file
    python code_eval.py "$split_file" $2 --artifact-name $1
done;
