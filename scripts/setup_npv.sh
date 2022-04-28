#!/bin/bash
echo pwd
echo ""
echo "===================================================="
echo "Making Raw Data"
python ./scripts/setup_datasets.py raw_npv --negation -n ${2:-4} -gen data/generated_test.json -rinputs 3


echo ""
echo "===================================================="
echo "Verifying Data"
python ./scripts/setup_datasets.py verify_npv --negation -n ${2:-4} -fratio ${1:-1}
