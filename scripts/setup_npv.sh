#!/bin/bash
echo ""
echo "===================================================="
echo "Making Raw Data"
python setup_datasets.py raw_npv --negation -n ${2:-4} -gen data/generated_test.json


echo ""
echo "===================================================="
echo "Verifying Data"
python setup_datasets.py verify_npv --negation -n ${2:-4} -fratio ${1:-1}
