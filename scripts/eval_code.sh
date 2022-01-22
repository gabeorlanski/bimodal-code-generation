#!/bin/bash
# Simple Experiment

python download_artifact.py $1 artifacts
python code_eval.py artifacts $2 --artifact-name $1

