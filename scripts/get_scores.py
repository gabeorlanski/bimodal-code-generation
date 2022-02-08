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


@click.command()
@click.argument("dump_path")
@click.argument("parsed_path")
def main(dump_path, parsed_path):
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
            stats[f"Posts in interval {str(interval)}"] = int(value)
        print(f"Stats for {f}:")
        for k, v in stats.items():
            print(f"\t{k:>16} = {v:0.3f}")
        all_stats[f] = stats

    parsed_path = PROJECT_ROOT.joinpath(parsed_path)
    for file in parsed_path.glob('*.jsonl'):
        print(f"Reading {file}")
        line_counts = 0
        question_scores = []
        all_answer_scores = []
        num_answers = []
        for line in map(json.loads, file.open('r')):
            question_scores.append(line['score'])
            answer_scores = [a['score'] for a in line['answers'].values()]
            num_answers.append(len(answer_scores))
            all_answer_scores.extend(answer_scores)

            line_counts += 1
            if line_counts % 5000 == 0:
                print(f"Finished line {line_counts}")

        question_scores = pd.Series(question_scores)
        all_answer_scores = pd.Series(all_answer_scores)
        num_answers = pd.Series(num_answers)
        all_stats[file.stem] = {
            "question_scores": question_scores.describe().to_dict(),
            "answer_scores"  : all_answer_scores.describe().to_dict(),
            "num_answers"    : num_answers.describe().to_dict()
        }
        all_stats[file.stem]['num_samples'] = line_counts
        all_stats[file.stem]['num_questions_with_answer'] = sum(num_answers>0)
    with dump_path.joinpath('stats.json').open('w') as f:
        json.dump(all_stats, f, indent=True)


if __name__ == '__main__':
    main()
