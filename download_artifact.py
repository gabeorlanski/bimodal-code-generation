import os

import wandb
import argparse
from pathlib import Path


def download_artifact(project, name, out):
    print(f"Downloading {name} to {out}")
    os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
    api = wandb.Api()
    artifact = api.artifact(f'{project}/{name}:latest')
    out_path = Path(out)
    artifact.download(str(out_path.resolve().absolute()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('artifact_name')
    parser.add_argument('output_dir')
    argv = parser.parse_args()
    download_artifact(argv.project, argv.artifact_name, argv.output_dir)
