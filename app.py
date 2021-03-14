# %%
import itertools
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from bokeh.plotting import figure, curdoc
from bokeh.models import *
from bokeh import layouts
from bokeh.layouts import layout
from bokeh.palettes import Category10_10 as palette

import tools
from tools import selected_rows, selected_row_single, selected_columns

import artifacts_dreamer2  # for handling app-specific artifacts
import artifacts_minigrid


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


def delete_run(run_id):
    with tools.Timer(f'mlflow.delete_run({run_id})', verbose=True):
        mlflow_client.delete_run(run_id)


def load_keys(runs_data=None):
    # runs_data = {'metrics.1': [run1val, run2val], 'metrics.2': [...], ...}
    if runs_data is None:
        return {'metric': [], 'value': []}
    metrics = {}
    for col in sorted(runs_data.keys()):
        if col.startswith('metrics.'):
            metrics_key = col.split('.')[1]
            vals = runs_data[col]
            if len(vals) > 0 and (vals[0] is not None) and (not np.isnan(vals[0])):
                metrics[metrics_key] = vals[0]  # Take first run value
    return {
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    }


def load_run_metrics(run=None, metric=DEFAULT_METRIC):
    if run is None:
        return tools.metrics_to_df([])
    with tools.Timer(f'mlflow.get_metric_history({metric})', verbose=True):
        hist = mlflow_client.get_metric_history(run['id'], metric)
    return tools.metrics_to_df(hist, run['name'])


def load_run_artifacts(run=None, path='d2_train_batch'):
    if run is None:
        return {}
    with tools.Timer(f'mlflow.list_artifacts()', verbose=True):
        artifacts = mlflow_client.list_artifacts(run['id'], path)
    return {
        'path': [f.path for f in artifacts],
        'file_size': [f.file_size for f in artifacts],
        'is_dir': [f.is_dir for f in artifacts],
    }


def load_artifact_steps(run_id, artifact_path):
    with tools.Timer(f'mlflow.download_artifact({artifact_path})', verbose=True):
        if artifact_path.endswith('.npz'):
            data = tools.download_artifact_npz(mlflow_client, run_id, artifact_path)
        else:
            raise NotImplementedError

    if artifact_path.startswith('d2_train_batch'):
        return artifacts_dreamer2.parse_d2_train_batch(data)
    else:
        raise NotImplementedError


def load_frame(step_data=None,
               image_keys=['image', 'image_rec', 'imag_image_1', 'imag_image_2']):

    if step_data is None:
        return {k: [] for k in image_keys}

    data = {}
    for k in image_keys:
        obs = step_data[k]
        assert obs.shape == (7, 7)  # Assuming MiniGrid
        img = artifacts_minigrid.render_obs(obs)
        img = tools.to_rgba(img)
        data[k] = [img]
    return data

# %%

# load_run_artifacts({'id':'db1a75611d464df08f1c7052cc8b1047'})
# data = download_artifact_npz('db1a75611d464df08f1c7052cc8b1047', 'd2_train_batch/0000951.npz')
# data['imag_image'].shape


# %%

def create_app(doc):

    # === Data sources ===

    runs_source = ColumnDataSource(data=load_runs())
    keys_source = ColumnDataSource(data=load_keys())
    metrics_sources = []
    for i in range(N_LINES):
        metrics_sources.append(ColumnDataSource(data=load_run_metrics()))

    artifacts_source = ColumnDataSource(data=load_run_artifacts())
    steps_source = ColumnDataSource(data={})
    frame_source = ColumnDataSource(data=load_frame())

    # Callbacks

    def refresh():
        print('Refreshing...')
        update_runs()
        update_metrics()

    def run_selected(attr, old, new):
        update_keys()
        update_metrics()
        update_artifacts()
    runs_source.selected.on_change('indices', run_selected)

    def key_selected(attr, old, new):
        update_metrics()
    keys_source.selected.on_change('indices', key_selected)

    def artifact_selected(attr, old, new):
        update_steps()
        update_frame()
    artifacts_source.selected.on_change('indices', artifact_selected)

    def step_selected(attr, old, new):
        update_frame()
    steps_source.selected.on_change('indices', step_selected)

    # Data update

    def update_runs():
        runs_source.data = load_runs()

    def delete_run_callback():
        runs = selected_rows(runs_source)
        if len(runs) == 1:
            delete_run(runs[0]['id'])
            update_runs()

    def update_keys():
        runs_data = selected_columns(runs_source)
        keys_source.data = load_keys(runs_data)

    def update_metrics():
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

    def update_artifacts():
        run = selected_row_single(runs_source)
        artifacts_source.data = load_run_artifacts(run)

    def update_steps():
        run = selected_row_single(runs_source)
        artifact = selected_row_single(artifacts_source)
        if run and artifact:
            steps_source.data = load_artifact_steps(run['id'], artifact['path'])
        else:
            steps_source.data = {}

    def update_frame():
        step = selected_row_single(steps_source)
        frame_source.data = load_frame(step)

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
        columns=[TableColumn(field="metric", title="metric"),
                 TableColumn(field="value", title="value", formatter=NumberFormatter(format="0.[000]")),
                 ],
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

    # === Artifacts ===

    # Artifacts list

    artifacts_table = DataTable(
        source=artifacts_source,
        columns=[TableColumn(field="path", title="path"),
                 ],
        width=200,
        height=600,
        selectable=True
    )

    # Artifact details

    fmt = NumberFormatter(format="0.[000]")
    artifact_steps_table = DataTable(
        source=steps_source,
        columns=[
            TableColumn(field="batch", formatter=fmt),
            TableColumn(field="step", formatter=fmt),
            TableColumn(field="action", formatter=fmt),
            TableColumn(field="reward", formatter=fmt),
            TableColumn(field="imag_action_1", formatter=fmt),
            TableColumn(field="imag_reward_1", formatter=fmt),
            TableColumn(field="imag_reward_2", formatter=fmt),
            TableColumn(field="imag_value_1", formatter=fmt),
            TableColumn(field="imag_target_1", formatter=fmt),
        ],
        width=600,
        height=600,
        selectable=True
    )

    w, h, dw, dh = 300, 300, 7, 7
    frame_figure_1 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None)
    frame_figure_2 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None)
    frame_figure_3 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None)
    frame_figure_4 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None)
    frame_figure_1.image_rgba(image='image', x=0, y=0, dw=dw, dh=dh, source=frame_source)
    frame_figure_2.image_rgba(image='image_rec', x=0, y=0, dw=dw, dh=dh, source=frame_source)
    frame_figure_3.image_rgba(image='imag_image_2', x=0, y=0, dw=dw, dh=dh, source=frame_source)
    frame_figure_4.image_rgba(image='imag_image_1', x=0, y=0, dw=dw, dh=dh, source=frame_source)

    # === Layout ===

    btn_delete = Button(label='Delete run', width=100)
    btn_delete.on_click(lambda _: delete_run_callback())

    if LIVE_REFRESH_SEC:
        doc.add_periodic_callback(refresh, LIVE_REFRESH_SEC * 1000)

    doc.add_root(
        layout([
            [runs_table, btn_delete],
            [Tabs(active=1, tabs=[
                Panel(title="Metrics", child=layout([
                    [keys_table, metrics_figure],
                ])),
                Panel(title="Artifacts", child=layout([
                    [artifacts_table, artifact_steps_table,
                        layout([
                            [frame_figure_1, frame_figure_2],
                            [frame_figure_3, frame_figure_4],
                        ])
                     ],
                ])),
            ])],
        ])
    )


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
