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


def load_artifacts(run=None, path=None, dirs=False):
    if run is None:
        return {}
    with tools.Timer(f'mlflow.list_artifacts({path})', verbose=True):
        artifacts = mlflow_client.list_artifacts(run['id'], path)
    artifacts = list([f for f in artifacts if f.is_dir == dirs])  # Filter dirs or files
    if not dirs:
        artifacts = list(reversed(artifacts))  # Order newest-first
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
            print(f'Artifact extension not supported: {artifact_path}')
            return {}

    if artifact_path.startswith('d2_train_batch/'):
        return artifacts_dreamer2.parse_d2_train_batch(data)
    elif artifact_path.startswith('d2_wm_predict/'):
        return artifacts_dreamer2.parse_d2_wm_predict(data)
    else:
        print(f'Artifact type not supported: {artifact_path}')
        return {}


def load_frame(step_data=None,
               image_keys=['image', 'image_rec', 'image_pred']):

    if step_data is None:
        return {k: [] for k in image_keys}

    data = {}
    for k in image_keys:
        if k in step_data:
            obs = step_data[k]
            assert obs.shape == (7, 7)  # Assuming MiniGrid
        else:
            obs = np.zeros((7, 7), dtype=int)
        img = artifacts_minigrid.render_obs(obs)
        img = tools.to_rgba(img)
        data[k] = [img]
    return data

# %%

# load_artifacts({'id':'db1a75611d464df08f1c7052cc8b1047'})
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

    artifacts_dir_source = ColumnDataSource(data=load_artifacts())
    artifacts_source = ColumnDataSource(data=load_artifacts())
    steps_source = ColumnDataSource(data={})
    frame_source = ColumnDataSource(data=load_frame())

    # Callbacks

    def refresh():
        print('Refreshing...')
        update_runs()
        # metrics
        update_keys()
        update_metrics()
        # artifacts
        update_artifacts()

    def run_selected(attr, old, new):
        # metrics
        update_keys()
        update_metrics()
        # artifacts
        update_artifacts_dir()
        update_artifacts()
        update_steps()
        update_frame()
    runs_source.selected.on_change('indices', run_selected)

    def key_selected(attr, old, new):
        update_metrics()
    keys_source.selected.on_change('indices', key_selected)

    def artifact_dir_selected(attr, old, new):
        update_artifacts()
        update_steps()
        update_frame()
    artifacts_dir_source.selected.on_change('indices', artifact_dir_selected)

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

    def update_artifacts_dir():
        run = selected_row_single(runs_source) if tabs.active == 1 else None  # Don't reload if another tab
        artifacts_dir_source.data = load_artifacts(run, dirs=True)

    def update_artifacts():
        run = selected_row_single(runs_source) if tabs.active == 1 else None  # Don't reload if another tab
        dir = selected_row_single(artifacts_dir_source)
        if run and dir:
            artifacts_source.data = load_artifacts(run, dir['path'])
        else:
            artifacts_source.data = {}

    def update_steps():
        run = selected_row_single(runs_source) if tabs.active == 1 else None  # Don't reload if another tab
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
            legend_field='metric',  # legend_label, legend_field, legend_group
            line_width=2,
            line_alpha=0.8)

    # === Artifacts ===

    # Artifacts list

    artifacts_dir_table = DataTable(
        source=artifacts_dir_source,
        columns=[TableColumn(field="path", title="directory")],
        width=200,
        height=100,
        selectable=True
    )

    artifacts_table = DataTable(
        source=artifacts_source,
        columns=[TableColumn(field="path", title="file")],
        width=200,
        height=500,
        selectable=True
    )

    # Artifact details

    fmt = NumberFormatter(format="0.[000]")
    artifact_steps_table = DataTable(
        source=steps_source,
        columns=[
            TableColumn(field="step", formatter=NumberFormatter(format="0,0")),
            TableColumn(field="action", title='action (last)', formatter=fmt),
            TableColumn(field="reward", title='reward (last)', formatter=fmt),
            #
            TableColumn(field="reward_rec", formatter=fmt),
            #
            TableColumn(field="action_pred", formatter=fmt),
            TableColumn(field="reward_pred", formatter=fmt),
            TableColumn(field="discount_pred", formatter=fmt),
            #
            TableColumn(field="value", formatter=fmt),
            TableColumn(field="value_target", formatter=fmt),
            TableColumn(field="loss_kl", title="kl", formatter=fmt),
        ],
        width=600,
        height=600,
        selectable=True
    )

    w, h, dw, dh = 300, 300, 7, 7
    frame_figure_1 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None, title='Observation')
    frame_figure_2 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None, title='Prediction')
    frame_figure_3 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None, title='Reconstruction')
    frame_figure_4 = fig = figure(plot_width=w, plot_height=h, x_range=[0, dw], y_range=[0, dh], toolbar_location=None, title='')
    frame_figure_1.image_rgba(image='image', x=0, y=0, dw=dw, dh=dh, source=frame_source)
    frame_figure_2.image_rgba(image='image_pred', x=0, y=0, dw=dw, dh=dh, source=frame_source)
    frame_figure_3.image_rgba(image='image_rec', x=0, y=0, dw=dw, dh=dh, source=frame_source)

    # === Layout ===

    btn_refresh = Button(label='Refresh', width=100)
    btn_refresh.on_click(lambda _: refresh())

    btn_delete = Button(label='Delete run', width=100)
    btn_delete.on_click(lambda _: delete_run_callback())

    tabs = Tabs(active=1, tabs=[
                Panel(title="Metrics", child=layout([
                    [keys_table, metrics_figure],
                ])),
                Panel(title="Artifacts", child=layout([
                    [
                        layouts.column([artifacts_dir_table, artifacts_table]), 
                        artifact_steps_table,
                        layout([
                            [frame_figure_1, frame_figure_2],
                            [frame_figure_3, frame_figure_4],
                        ])
                     ],
                ])),
                ])

    doc.add_root(
        layout([
            [runs_table, layouts.column([btn_refresh, btn_delete])],
            [tabs],
        ])
    )


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
