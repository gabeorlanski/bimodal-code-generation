#!/bin/bash

make build-docker
docker run --rm --name code_eval go-adversarial-code:latest bash scripts/eval_code.sh $1 $2