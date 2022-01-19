#!/bin/bash
# Run the code eval in the docker container
docker run --name code_eval go-adversarial-code:latest python code_eval.py $1 $2 ${3:-''}
docker kill code_eval
docker rm code_eval
