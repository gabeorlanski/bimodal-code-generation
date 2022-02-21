from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys

try:
    from src.common import PROJECT_ROOT
except ImportError:
    if str(Path(__file__).parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.common import PROJECT_ROOT

OVR_PCT_COLUMNS = ["test/syntax_error_pct_ovr", "test/runtime_error_pct_ovr",
                   "test/failed_tests_pct_ovr", "test/correct_pct_ovr"]


def multi_figure_bar_graph(
        graph_dict: Dict[str, Dict],
        color_mapping: Dict,
        ordering: List = None
) -> List:
    if not ordering:
        ordering = list(graph_dict)
    bars = []
    for i, r in enumerate(ordering):
        graph_data = graph_dict[r]
        graph_x = list(graph_data)
        graph_y = list(graph_data.values())
        bars.append(go.Bar(
            name=str(r),
            x=graph_x,
            y=graph_y,
            marker_color=color_mapping[r],
            showlegend=True,
            texttemplate='%{y:.2f}', textposition='inside'
        ))
    return bars


if __name__ == '__main__':
    import plotly.express as px

    main_df = pd.read_json(PROJECT_ROOT.joinpath('data', 'run_data', 'MBPP[execution].jsonl'))
    print(f"Testing Visualization with {len(main_df)} elements")

    met_name_map = {
        "test/syntax_error_pct_ovr" : {
            "name" : "Syntax Error",
            "color": "#990000"
        },
        "test/runtime_error_pct_ovr": {
            "name" : "Runtime Error",
            "color": "#FFA500"
        },
        "test/correct_pct_ovr"      : {
            "name" : "Correct",
            "color": "#5bbd4a"
        },
        "test/failed_tests_pct_ovr" : {
            "name" : "Failed Tests",
            "color": "#ff6666"
        }
    }
    if not PROJECT_ROOT.joinpath('imgs').exists():
        PROJECT_ROOT.joinpath('imgs').mkdir(parents=True)

    run_colors = px.colors.qualitative.Set2
    for dump in main_df['meta.ablation_vals.DumpName'].unique():
        print(dump)
        if dump == "Baseline":
            continue

        plot_df = main_df[main_df.name.str.contains(f"FullData.ParrotSmall.{dump}")][
            ["name", 'meta.ablation_vals.AnswerCount', *OVR_PCT_COLUMNS]]
        plot_df = plot_df.rename(columns={"meta.ablation_vals.AnswerCount": "answer_count"})
        runs = plot_df['answer_count'].tolist()
        runs = list(
            sorted(
                runs,
                key=lambda f: int(f) if f != 'All' else float('inf')
            )
        )
        col_graphs = defaultdict(dict)
        for i, r in plot_df.iterrows():
            col_graphs[r['answer_count']] = {}
            for met in OVR_PCT_COLUMNS:
                col_graphs[r['answer_count']][met_name_map[met]['name']] = r[met]

        fig = go.Figure(data=multi_figure_bar_graph(
            col_graphs,
            color_mapping={k: run_colors[i] for i, k in enumerate(runs)},
            ordering=runs
        ))
        fig.update_layout(
            title=f"{dump} Filtered Data",
            title_x=0.5,
            xaxis_title="Error Type",
            yaxis_title="% Of Total Predictions",
            height=400,
            width=720,
            legend_title_text="# Answers"
        )
        fig.write_image(str(PROJECT_ROOT.joinpath('imgs', f"{dump}_errors.png")))
        # fig.show()
