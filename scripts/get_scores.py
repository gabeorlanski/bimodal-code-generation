import pandas as pd
import click
from pathlib import Path
import sys

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging, flatten
from src.config.experiments import load_composed_experiments_from_file
from src.evaluation.code_eval import get_metrics_from_list


@click.command()
@click.argument("dump_path")
def main(dump_path):
    dump_path = PROJECT_ROOT.joinpath(dump_path)
    files = [
        'valid_answer_scores',
        'valid_question_scores',
        'all_answer_scores',
        'all_question_scores'
    ]
    for f in files:
        path_to_file = dump_path.joinpath(f"{f}.txt")
        print(f"Reading {path_to_file}")
        stats = get_metrics_from_list(
            '',
            list(map(int, path_to_file.read_text().splitlines(False)))
        )
        print(f"Stats for {f}:")
        for k, v in stats.items():
            print(f"\t{k:>16} = {v:0.3f}")


if __name__ == '__main__':
    main()