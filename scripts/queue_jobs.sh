#!/bin/bash
# Queue Slurm Jobs

sbatch -J 'high_qual' make_splits.sbatch "python scripts/parse_so_data.py -out data/parsed_so parse data/stack_exchange/stackoverflow 32 high_qual -min '30' -val 0.01"
sbatch -J 'general_parsing' make_splits.sbatch "python scripts/parse_so_data.py -out data/parsed_so parse data/stack_exchange/stackoverflow 32 general -val 0.01"
sbatch -J 'exception' make_splits.sbatch "python scripts/parse_so_data.py -out data/parsed_so parse data/stack_exchange/stackoverflow 32 exceptions -contains exception,error,trace"
