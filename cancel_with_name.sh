#!/bin/bash

files=(find  -maxdepth 1 -name "${1}*.sh")
for f in generated_experiments/cancel_scripts/*.sh; do
  if [[ "$f" == *"$1"* ]]; then
    echo "$f"
  fi
done