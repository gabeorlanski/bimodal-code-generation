#!/bin/bash
#SBATCH --output=./logs/sodownload.out
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -c 1

cd data/SEDumps
wget https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z
