#!/bin/bash
echo ""
echo "===================================================="
echo "Making Raw Data"
python ./scripts/setup_datasets.py --seed ${5:-1999} raw_npv test --negation -n ${2:-4} \
  -gen data/generated_test.json -gmod ${3:-1.5}
python ./scripts/setup_datasets.py --seed ${5:-1999} raw_npv train --negation -n ${2:-4} \
  -gen data/generated_test.json -gmod ${3:-1.5}
python ./scripts/setup_datasets.py --seed ${5:-1999} raw_npv validation --negation -n ${2:-4} \
  -gen data/generated_test.json -gmod ${3:-1.5}


echo ""
echo "===================================================="
echo "Verifying Data"
python ./scripts/setup_datasets.py --seed ${5:-1999} verify_npv test --negation -n ${2:-4} \
  -fratio ${1:-1} -gratio ${4:-1}
python ./scripts/setup_datasets.py --seed ${5:-1999} verify_npv train --negation -n ${2:-4} \
  -fratio ${1:-1} -gratio ${4:-1}
python ./scripts/setup_datasets.py --seed ${5:-1999} verify_npv validation --negation -n ${2:-4} \
  -fratio ${1:-1} -gratio ${4:-1}
