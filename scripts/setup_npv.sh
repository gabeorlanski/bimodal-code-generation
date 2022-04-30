#!/bin/bash
echo ""
echo "===================================================="
echo "Making Raw Data"
python ./scripts/setup_datasets.py raw_npv test --negation -n ${2:-4} -gen data/generated_test.json
python ./scripts/setup_datasets.py raw_npv train --negation -n ${2:-4}
python ./scripts/setup_datasets.py raw_npv validation --negation -n ${2:-4}


echo ""
echo "===================================================="
echo "Verifying Data"
python ./scripts/setup_datasets.py verify_npv test --negation -n ${2:-4} -fratio ${1:-1}
python ./scripts/setup_datasets.py verify_npv train --negation -n ${2:-4} -fratio ${1:-1}
python ./scripts/setup_datasets.py verify_npv validation --negation -n ${2:-4} -fratio ${1:-1}
