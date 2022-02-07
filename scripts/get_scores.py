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
    all_stats = {}
    for f in files:
        path_to_file = dump_path.joinpath(f"{f}.txt")
        print(f"Reading {path_to_file}")
        scores = list(map(int, path_to_file.read_text().splitlines(False)))

        scores_series = pd.Series(scores)
        stats: Dict = scores_series.describe(  # type:ignore
            percentiles=[i / 100 for i in range(5, 100, 5)]).to_dict()

        bins = [stats['min']]

        for i in range(5, 100, 5):
            if stats[f"{i}%"] not in bins:
                bins.append(stats[f"{i}%"])
        bins.append(stats['max'])
        score_bins = scores_series.groupby(pd.cut(scores_series, bins)).agg(['count'])
        for interval, value in score_bins['count'].iteritems():
            stats[str(interval)] = int(value)
        print(f"Stats for {f}:")
        for k, v in stats.items():
            print(f"\t{k:>16} = {v:0.3f}")
        all_stats[f] = stats

    with dump_path.joinpath('stats.json').open('w') as f:
        json.dump(all_stats, f, indent=True)


if __name__ == '__main__':
    main()
