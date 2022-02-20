import json
from collections import defaultdict
import argparse
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
import shutil
import os
import click

PROJECT_ROOT = Path.cwd()
if 'scripts' in str(PROJECT_ROOT):
    while 'scripts' in str(PROJECT_ROOT):
        PROJECT_ROOT = PROJECT_ROOT.parent


@click.command()
def download_wandb_runs():
    out_path = PROJECT_ROOT.joinpath('data/run_data')
    if not out_path.exists():
        out_path.mkdir(parents=True)

    print(f'Downloading to {out_path}')
    api = wandb.Api()

    print(f"Downloading runs")

    # Project is specified by <entity/project-name>
    runs = api.runs("nyu-code-research/adversarial-code")
    print(f"{len(runs)} runs found")
    summary_keys = set()
    records = defaultdict(list)
    for run in tqdm(runs, desc='Downloading'):
        record_dict = {}
        run_summary = {k: v for k, v in run.summary._json_dict.items()
                       if not k.startswith('_')}

        if "OLD_" in run.group:
            continue

        summary_keys.update(set(run_summary))
        record_dict.update(run_summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        record_dict.update({k: v for k, v in run.config.items()
                            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        record_dict['name'] = run.name
        record_dict['group'] = run.group
        records[run.group].append(record_dict)

    for group_name, group_records in records.items():
        print(f"Saving {group_name}")
        runs_df = pd.DataFrame.from_records(group_records)
        runs_df.to_json(out_path.joinpath(f"{group_name}.jsonl"),orient='records')


if __name__ == '__main__':
    download_wandb_runs()
