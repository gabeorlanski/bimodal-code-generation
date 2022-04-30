#!/bin/bash
echo ""
echo "===================================================="
echo "Making Raw Data"
python ./scripts/setup_datasets.py raw_npv $1 --negation -n ${3:-4} -gen data/generated_test.json


echo ""
echo "===================================================="
echo "Verifying Data"
python ./scripts/setup_datasets.py verify_npv $1 --negation -n ${3:-4} -fratio ${2:-1}
