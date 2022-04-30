#!/bin/bash
echo ""
echo "===================================================="
echo "Making Raw Data"
python ./scripts/setup_datasets.py raw_npv --seed ${3:-1999} test --negation -n ${2:-4} -gen data/generated_test.json
python ./scripts/setup_datasets.py raw_npv --seed ${3:-1999} train --negation -n ${2:-4} -gen data/generated_test.json
python ./scripts/setup_datasets.py raw_npv --seed ${3:-1999} validation --negation -n ${2:-4} -gen data/generated_test.json


echo ""
echo "===================================================="
echo "Verifying Data"
python ./scripts/setup_datasets.py --seed ${3:-1999} verify_npv test --negation -n ${2:-4} -fratio ${1:-1}
python ./scripts/setup_datasets.py --seed ${3:-1999} verify_npv train --negation -n ${2:-4} -fratio ${1:-1}
python ./scripts/setup_datasets.py --seed ${3:-1999} verify_npv validation --negation -n ${2:-4} -fratio ${1:-1}
