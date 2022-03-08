from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

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


def multi_line_plot(
        df,
        filter_fn,
        name_col,
        value_columns,
        x_values,
        name_remap_dict=None,
        ordering=None,
        dashed=None,
        line_dict=None,
        color_palette=None
):
    traces = []

    global_line_dict = line_dict or {}
    dashed = dashed or []
    color_palette=color_palette or px.colors.qualitative.T10

    filter_mask = df.apply(filter_fn, axis=1)
    filtered_df = df[filter_mask]
    if ordering is None:
        ordering = sorted(filtered_df[name_col].values.tolist())

    for i, value in enumerate(ordering):
        row = filtered_df[filtered_df[name_col] == value].iloc[0]
        y_values = row[value_columns].values.tolist()
        name_use = row[name_col]
        if name_remap_dict is not None:
            name_use = name_remap_dict.get(name_use, name_use)

        color = color_palette[i]

        line_dict = {
            "color": color,
            **global_line_dict
        }

        if value in dashed:
            line_dict['dash'] = 'dash'

        traces.append(go.Scatter(
            x=x_values,
            y=y_values,
            name=name_use,
            mode='lines',
            line=line_dict,
            showlegend=True
        ))
        traces.append(go.Scatter(
            x=x_values,
            y=y_values,
            name=name_use,
            mode='markers',
            marker={
                "color": color,
            },
            showlegend=False
        ))
    return traces


def main():
    import plotly.express as px

    main_df = pd.read_json(PROJECT_ROOT.joinpath('data', 'run_data', 'MBPP[execution].jsonl'))
    main_df['full_name'] = main_df.name.copy()
    main_df['name'] = main_df['name'].apply(lambda n: n.split('-')[0])
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
    pass_at_k_vals = [1, 5, 10, 25, 50, 100]
    runs_to_keep = [
        'PythonNoNL',
        'PythonNoCode',
        'PythonTitleShuffle'
    ]
    x = multi_line_plot(
        main_df,
        filter_fn=lambda r: (
                (r['meta.ablation'] in runs_to_keep and '512' not in r['name'])
                or r['name'] == "CodeParrotSmall"
        ),
        name_col='meta.ablation',
        value_columns=[f'test/pass@{k}' for k in pass_at_k_vals],
        x_values=pass_at_k_vals
    )
    fig = make_subplots(1,2,subplot_titles=("TEST","TEST"))
    for t in x:
        fig.add_trace(t,row=1,col=1)
    for anno in fig['layout']['annotations']:
        print("?")
    fig.show()
    print("???")


if __name__ == '__main__':
    main()
