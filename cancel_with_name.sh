#!/bin/bash

for f in generated_experiments/cancel_scripts/*.sh; do
#  if grep -q "$f" <<< "${1}"; then
  if [[ "$f" == *"$1"* ]]; then
    echo "${f}"
    bash $f
  fi
done