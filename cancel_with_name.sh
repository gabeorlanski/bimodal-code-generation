#!/bin/bash

files=(find generated_experiments/cancel_scripts/ -maxdepth 1 -name "${1}*.sh")
for f in $files; do
  echo "$f"
done