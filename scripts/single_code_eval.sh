#!/bin/bash
# Run the code eval in the docker container

make build-docker
docker run --rm --name code_eval go-adversarial-code:latest python code_eval.py $1 $2 -notrack