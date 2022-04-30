#!/bin/bash
echo ""
echo "===================================================="
echo "Making Raw Data"
python ./scripts/setup_datasets.py $1 test --negation -n ${3:-4} -gen data/generated_test.json


echo ""
echo "===================================================="
echo "Verifying Data"
python ./scripts/setup_datasets.py verify_npv test --negation -n ${3:-4} -fratio ${2:-1}
