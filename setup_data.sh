#!/bin/bash
# Get the data

if [[ ! -d data ]]
then
    mkdir data
fi

# TODO: Update this when done
## Setup the APPS dataset
#python src/data/apps/apps_create_split.py data

# Setup the MBPP data
echo "\n\nSetting up the MBPP Dataset"
echo "=============================================\n"
if [[ ! -d data/MBPP ]]
then
    mkdir data/MBPP
fi
wget https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
mv mbpp.jsonl data/MBPP/mbpp.jsonl
wget https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json
mv sanitized-mbpp.json data/MBPP/sanitized-mbpp.json
python ./src/data/mbpp.py data/MBPP