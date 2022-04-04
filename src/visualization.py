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
import seaborn as sns
import sys
import dataframe_image as dfi

try:
    from src.common import PROJECT_ROOT
except ImportError:
    if str(Path(__file__).parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).parents[1]))
    from src.common import PROJECT_ROOT

OVR_PCT_COLUMNS = ["test/syntax_error_pct_ovr", "test/runtime_error_pct_ovr",
                   "test/failed_tests_pct_ovr", "test/correct_pct_ovr"]

PASS_AT_K_COL = [f'pass@{k}' for k in [1, 5, 10, 25, 50, 100]]


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
    color_palette = color_palette or px.colors.qualitative.T10

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


def make_gradient_table(
        raw_df,
        columns_keep,
        gradient_stat_columns,
        float_columns=None,
        rename_columns=None,
        formatting_cols=None,
        no_grad_rows=None,
        cmap=None
):
    rename_columns = rename_columns or {}
    float_columns = float_columns or []
    col_to_keep = [*list(rename_columns), *columns_keep,*float_columns,*gradient_stat_columns]

    filtered_df = raw_df[col_to_keep]
    if no_grad_rows:
        grad_index = filtered_df[~filtered_df['display_name'].isin(no_grad_rows)].index
    else:
        grad_index = filtered_df.index
    filtered_df = filtered_df[col_to_keep].rename(columns=rename_columns)

    styled_df = filtered_df.style.format(
        precision=3,
        na_rep='MISSING',
        thousands=" ",
        subset=gradient_stat_columns+float_columns
    )
    for c, format_dict in (formatting_cols or {}).items():
        styled_df = styled_df.set_properties(
            subset=[c] if not isinstance(c, (dict, tuple)) else c,
            **format_dict
        )
    if cmap is None:
        cmap = sns.diverging_palette(220, 20, as_cmap=True)

    styled_df = styled_df.set_properties(**{
        'font-size': '11pt',
    })
    styled_df = styled_df.background_gradient(
        cmap=cmap,
        axis=0,
        subset=(grad_index, gradient_stat_columns)
    )
    return styled_df


def make_minmax_highlight_table(
        df,
        columns_keep,
        stat_columns,
        rename_columns=None,
        formatting_cols=None,
        no_grad_rows=None
):
    rename_columns = rename_columns or {}

    filtered_df = df[[*list(rename_columns), *columns_keep, *stat_columns]]
    if no_grad_rows:
        grad_index = filtered_df[~filtered_df['display_name'].isin(no_grad_rows)].index
    else:
        grad_index = filtered_df.index
    filtered_df = filtered_df.rename(columns=rename_columns)

    styled_df = filtered_df.style.format(
        precision=3,
        na_rep='MISSING',
        thousands=" ",
        subset=stat_columns
    )
    for c, format_dict in (formatting_cols or {}).items():
        styled_df = styled_df.set_properties(
            subset=[c] if not isinstance(c, (dict, tuple)) else c,
            **format_dict
        )

    def highlight(s, max_props='', min_props=''):
        out = []
        for r in s:
            if r == s.max():
                out.append(max_props)
            elif r == s.min():
                out.append(min_props)
            else:
                out.append('')
        return out

    styled_df = styled_df.apply(
        highlight,
        max_props='background-color: #70db70;',
        min_props='background-color: #ffad99;',
        axis=0,
        subset=(grad_index, stat_columns)
    )
    return styled_df


def fix_name(c_name):
    out_name = c_name.split('-')[0].split('_')
    if len(out_name) == 1:
        return out_name[0]
    return '_'.join(out_name[:-1])


def prepare_df_pass_at_k(df, runs_to_keep=None):
    pass_at_k_cols = {
        f'test/{k}': k for k in PASS_AT_K_COL
    }
    if runs_to_keep:
        out = df[df.name.isin(runs_to_keep)].copy()
    else:
        out = df.copy()
    out = out.rename(columns=pass_at_k_cols)

    if runs_to_keep:
        out['display_name'] = out['name'].apply(lambda n: runs_to_keep[n])

    return out


def get_pass_at_k_df(df, extra_columns=None):
    extra_columns = extra_columns or []
    col_to_keep = ['meta.ablation', 'name', 'meta.card_name']
    if 'display_name' in df.columns:
        col_to_keep.append('display_name')
    col_to_keep.extend(extra_columns)

    return df[col_to_keep + PASS_AT_K_COL]


def save_style_df(style_df, name, path=None):
    if isinstance(path, str):
        save_path = Path(path)
    else:
        save_path = path or Path()
    dfi.export(style_df, str(save_path.joinpath(f"{name}.png")))


def main():
    main_df = pd.read_json(PROJECT_ROOT.joinpath('data', 'run_data', 'MBPP[execution].jsonl'))
    main_df['full_name'] = main_df.name.copy()
    main_df['name'] = main_df['name'].apply(fix_name)
    print(f"Testing Visualization with {len(main_df)} elements")

    runs_to_keep = {
        f'GPTNeo125M.Eval': 'GPT Neo 125M',
        f'CodeParrotSmall': 'CodeParrot Small'
    }

    raw_df = prepare_df_pass_at_k(main_df[main_df.name.isin(runs_to_keep)])
    raw_df = get_pass_at_k_df(raw_df)
    make_gradient_table(raw_df,
                        [],
                        ['pass@1','pass@10','pass@100'])
    print(len(raw_df))


if __name__ == '__main__':
    main()
