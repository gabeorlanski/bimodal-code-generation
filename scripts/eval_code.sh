#!/bin/bash
# Simple Experiment

python download_artifact.py $1 $2 artifacts
python code_eval.py artifacts $3 --artifact-name $2

