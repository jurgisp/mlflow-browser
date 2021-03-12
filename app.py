# %%
import itertools
import mlflow
from mlflow.tracking import MlflowClient
from numpy.lib.function_base import select
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import *
from bokeh import layouts
from bokeh.layouts import layout
from bokeh.palettes import Category10_10 as palette

import tools
from tools import selected_rows

N_LINES = 10
MAX_RUNS = 100
LIVE_REFRESH_SEC = 30
DEFAULT_METRIC = '_loss'

mlflow_client = MlflowClient()


def load_runs():
    with tools.Timer(f'mlflow.search_runs()', verbose=True):
        df = mlflow.search_runs(max_results=MAX_RUNS)
    df['id'] = df['run_id']
    df['name'] = df['tags.mlflow.runName']
    return df


def load_keys(runs_data):
    run_columns = runs_data.keys()
    metric_columns = [c for c in run_columns if c.startswith('metrics.')]
    metric_keys = [c.split('.')[1] for c in metric_columns]
    metric_keys.sort()
    return {'metric': metric_keys}


def load_run_metrics(run=None, metric=DEFAULT_METRIC):
    if run is None:
        return tools.metrics_to_df([])
    with tools.Timer(f'mlflow.get_metric_history({metric})', verbose=True):
        hist = mlflow_client.get_metric_history(run['id'], metric)
    return tools.metrics_to_df(hist, run['name'])


# %%

# df = load_runs()
# list(df.columns)

# %%


def create_app(doc):

    # === Data sources ===

    runs_source = ColumnDataSource(data=load_runs())
    keys_source = ColumnDataSource(data=load_keys(runs_source.data))
    metrics_sources = []
    for i in range(N_LINES):
        metrics_sources.append(ColumnDataSource(data=load_run_metrics()))

    # Callbacks

    runs_source.selected.on_change('indices', lambda attr, old, new: reload_metrics())
    keys_source.selected.on_change('indices', lambda attr, old, new: reload_metrics())

    def reload_runs():
        runs_source.data = load_runs()

    def reload_metrics():
        runs = selected_rows(runs_source)
        keys = selected_rows(keys_source)
        if len(keys) > 0:
            keys = [row['metric'] for row in keys]
        else:
            keys = [DEFAULT_METRIC]

        run_keys = list(itertools.product(runs, keys))

        for i in range(N_LINES):
            if i < len(run_keys):
                metrics_sources[i].data = load_run_metrics(run_keys[i][0], metric=run_keys[i][1])
            else:
                metrics_sources[i].data = load_run_metrics()

    def refresh():
        print('Refreshing...')
        reload_runs()
        reload_metrics()

    # === Layout ===

    # Runs table

    runs_table = DataTable(
        source=runs_source,
        columns=[TableColumn(field="name", title="run"),
                 TableColumn(field="start_time", title="time", formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
                 TableColumn(field="metrics._step", title="step", formatter=NumberFormatter(format="0,0")),
                 TableColumn(field="metrics._loss", title="loss", formatter=NumberFormatter(format="0.00")),
                 TableColumn(field="metrics.actor_ent", title="actor_ent", formatter=NumberFormatter(format="0.00")),
                 TableColumn(field="metrics.train_return", title="train_return", formatter=NumberFormatter(format="0.00")),
                 ],
        width=1200,
        height=250,
        selectable=True
    )

    # Keys table

    keys_table = DataTable(
        source=keys_source,
        columns=[TableColumn(field="metric", title="metric")],
        width=200,
        height=600,
        selectable=True
    )

    # Metrics figure

    # TODO: multiline+legend https://github.com/bokeh/bokeh/pull/8218
    metrics_figure = figure(
        x_axis_label='step',
        y_axis_label='value',
        plot_width=1200,
        plot_height=600,
        tooltips=[
            ("run", "@run"),
            ("metric", "@metric"),
            ("step", "@step"),
            ("value", "@value"),
        ]
    )
    metrics_figure.toolbar.active_scroll = metrics_figure.select_one(WheelZoomTool)
    for i in range(N_LINES):
        metrics_figure.line(
            x='step',
            y='value',
            source=metrics_sources[i],
            color=palette[i],
            # legend_field='run',  # legend_label, legend_field, legend_group
            line_width=2,
            line_alpha=0.8)

    # Layout

    if LIVE_REFRESH_SEC:
        doc.add_periodic_callback(refresh, LIVE_REFRESH_SEC * 1000)

    doc.add_root(
        layout([
            [runs_table],
            [keys_table, metrics_figure],
        ])
    )


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
