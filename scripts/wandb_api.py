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


@click.group()
@click.pass_context
def cli(ctx):
    os.environ["WANDB_API_KEY"] = PROJECT_ROOT.joinpath('wandb_secret.txt').read_text().strip()
    ctx.obj = {
        'entity' : 'nyu-code-research',
        'project': 'so-code-gen'
    }


@cli.command('upload_model')
@click.argument('model_directory', metavar="<Model Directory or Checkpoint Dir>")
@click.argument('run_id', metavar="<The Run ID to add the model too>")
@click.pass_context
def upload_model(ctx, model_directory, run_id):
    model_directory = PROJECT_ROOT.joinpath(model_directory)
    print(f"Uploading {model_directory} to {run_id}")

    run = wandb.init(  # type: ignore
        entity=ctx.obj['entity'],
        project=ctx.obj['project'],
        id=run_id,
        resume=True
    )

    print("Loaded run, now making artifact")
    artifact = wandb.Artifact(name=f"model-{run.id}", type="model")  # type: ignore
    artifact.add_dir(str(model_directory.resolve().absolute()))
    run.log_artifact(artifact)
    run.finish()


@cli.command('download')
@click.pass_context
def download_wandb_runs(ctx):
    out_path = PROJECT_ROOT.joinpath('data/run_data')
    if not out_path.exists():
        out_path.mkdir(parents=True)

    print(f'Downloading to {out_path}')
    api = wandb.Api()  # type: ignore

    print(f"Downloading runs")

    # Project is specified by <entity/project-name>
    runs = api.runs(f"{ctx.obj['entity']}/{ctx.obj['project']}")
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
        runs_df.to_json(out_path.joinpath(f"{group_name}.jsonl"), orient='records')


if __name__ == '__main__':
    cli()
