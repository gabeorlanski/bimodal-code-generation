#!/bin/bash
#SBATCH --output=./logs/job_parse_so.log
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH -c 48
echo "Starting Parse"