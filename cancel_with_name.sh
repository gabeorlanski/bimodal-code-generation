#!/bin/bash

for f in generated_experiments/cancel_scripts/*.sh; do
  echo "$f ${1}"
  if grep -q "$f" <<< "${1}"; then
    echo "$f"
  fi
done