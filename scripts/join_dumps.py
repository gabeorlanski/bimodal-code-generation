import json
from typing import Dict

import pandas as pd
import click
from pathlib import Path
import sys

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT
import numpy as np


@click.command()
@click.argument("dump_name")
@click.argument("parsed_path")
@click.argument("out_path")
@click.option('--val', 'val_size', type=int, default=500)
def main(dump_name, parsed_path, out_path, val_size):
    if parsed_path == out_path:
        raise ValueError("Parsed and out cannot be the same")
    print(f"Joining {dump_name}:")
    path_to_train = PROJECT_ROOT.joinpath(parsed_path,f"{dump_name}.jsonl")
    path_to_val = PROJECT_ROOT.joinpath(parsed_path,f"{dump_name}_val.jsonl")
    print(f"{path_to_train=}")
    print(f"{path_to_val=}")

    if not PROJECT_ROOT.joinpath(out_path).exists():
        PROJECT_ROOT.joinpath(out_path).mkdir(parents=True)
    out_file = PROJECT_ROOT.joinpath(out_path).joinpath(f"{dump_name}.jsonl").open('w')
    out_val_file = PROJECT_ROOT.joinpath(out_path).joinpath(f"{dump_name}_val.jsonl").open('w')
    rng = np.random.default_rng(1)
    samples_in_val = list(map(json.loads, path_to_val.read_text().splitlines()))
    sample_mask = np.zeros((len(samples_in_val),), dtype=bool)
    sample_mask[rng.choice(len(samples_in_val), (min(val_size, len(samples_in_val)),),
                           replace=False)] = True
    for i, v in enumerate(samples_in_val):
        if sample_mask[i]:
            out_val_file.write(json.dumps(v) + '\n')
        else:
            out_file.write(json.dumps(v) + '\n')

    out_val_file.close()
    line_num = 0
    for line in path_to_train.read_text().splitlines(False):
        out_file.write(line+'\n')

        line_num += 1
        if line_num % 5000 == 0:
            print(f"Finished {line_num}")
    out_file.close()


if __name__ == '__main__':
    main()
