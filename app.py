# %%
import itertools
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

import bokeh.plotting
from bokeh.plotting import curdoc
from bokeh.models import *
from bokeh import layouts
from bokeh.layouts import layout
from bokeh.palettes import Category10_10 as palette

import tools
from tools import selected_rows, selected_row_single, selected_columns

import artifacts_dreamer2  # for handling app-specific artifacts
import artifacts_minigrid


MAX_RUNS = 100
DEFAULT_METRIC = '_loss'

PLAY_INTERVAL = 500
PLAY_DELAY = 5000

mlflow_client = MlflowClient()


def figure(tools='box_select,tap,wheel_zoom,reset', active_scroll=True, hide_axes=False, **kwargs):
    fig = bokeh.plotting.figure(
        tools=tools,
        **kwargs,
    )
    if active_scroll:
        fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)
    if hide_axes:
        fig.xaxis.visible = False
        fig.yaxis.visible = False
    return fig


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


def load_run_metrics(runs=[], metrics=[DEFAULT_METRIC]):
    data = []
    i = 0
    for run in runs:
        run_id, run_name = run['id'], run['name']
        for metric in metrics:
            with tools.Timer(f'mlflow.get_metric_history({metric})', verbose=True):
                hist = mlflow_client.get_metric_history(run_id, metric)
            data.append([
                run_name,
                metric,
                f'{metric} [{run_name}]' if len(runs) > 1 else f'{metric}',
                palette[i % len(palette)],
                np.array([m.timestamp for m in hist]),
                np.array([m.step for m in hist]),
                np.array([m.value for m in hist]),
            ])
            i += 1
    df = pd.DataFrame(data, columns=[
        'run',
        'metric',
        'legend',
        'color',
        'timestamps',
        'steps',
        'values'
        ])
    return df


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

    if artifact_path.startswith('d2_wm_predict'):
        return artifacts_dreamer2.parse_d2_wm_predict(data)

    if artifact_path.startswith('d2_train_episodes/') or artifact_path.startswith('d2_eval_episodes/'):
        return artifacts_dreamer2.parse_d2_episodes(data)

    print(f'Artifact type not supported: {artifact_path}')
    return {}


def load_frame(step_data=None,
               image_keys=['image', 'image_rec', 'image_pred', 'map', 'map_agent', 'map_rec', 'map_rec_global']
               ):
    if step_data is None:
        return {k: [] for k in image_keys}

    sd = step_data
    if 'map_rec' in sd and 'map_agent' in sd and sd['map_rec'].shape[0] > sd['map_agent'].shape[0]:
        # map_rec is agent-centric
        # transform it for easier viewing
        map_agent = artifacts_minigrid.CAT_TO_OBJ[sd['map_agent']]
        agent_pos, agent_dir = artifacts_minigrid._get_agent_pos(map_agent)
        sd['map_rec_global'] = artifacts_minigrid._map_centric_to_global(sd['map_rec'], agent_pos, agent_dir, map_agent.shape[:2])

    data = {}
    for k in image_keys:
        obs = sd.get(k)
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
    metrics_source = ColumnDataSource(data=load_run_metrics())

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

    def play_frame():
        ix = steps_source.selected.indices
        if len(ix) == 1:
            steps_source.selected.indices = [ix[0] + 1]
        else:
            steps_source.selected.indices = [0]

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
        metrics_source.data = load_run_metrics(runs, keys)

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
        width=300,
        height=600,
        selectable=True
    )

    # Metrics figure

    metrics_figure = figure(
        x_axis_label='step',
        y_axis_label='value',
        plot_width=1000,
        plot_height=600,
        tooltips=[
            ("run", "@run"),
            ("metric", "@metric"),
            ("step", "@step"),
            ("value", "@value"),
        ],
    )
    metrics_figure.multi_line(
        xs='steps',
        ys='values',
        source=metrics_source,
        color='color',
        legend_field='legend',
        line_width=2,
        line_alpha=0.8)

    # === Artifacts ===

    # Artifacts list

    artifacts_dir_table = DataTable(
        source=artifacts_dir_source,
        columns=[TableColumn(field="path", title="directory")],
        width=200,
        height=150,
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

    kwargs = dict(plot_width=250, plot_height=250, x_range=[0, 10], y_range=[0, 10], toolbar_location=None, active_scroll=False, hide_axes=True)
    frame_figure_1 = fig = figure(title='Observation', **kwargs)
    frame_figure_2 = fig = figure(title='Prediction', **kwargs)
    frame_figure_3 = fig = figure(title='Reconstruction', **kwargs)
    frame_figure_4 = fig = figure(title='Map', **kwargs)
    frame_figure_5 = fig = figure(title='Map prediction (global)', **kwargs)
    frame_figure_6 = fig = figure(title='Map prediction', **kwargs)
    kwargs = dict(x=0, y=0, dw=10, dh=10)
    frame_figure_1.image_rgba(image='image', source=frame_source, **kwargs)
    frame_figure_2.image_rgba(image='image_pred', source=frame_source, **kwargs)
    frame_figure_3.image_rgba(image='image_rec', source=frame_source, **kwargs)
    frame_figure_4.image_rgba(image='map_agent', source=frame_source, **kwargs)
    # frame_figure_5.image_rgba(image='map', source=frame_source, **kwargs)
    frame_figure_5.image_rgba(image='map_rec_global', source=frame_source, **kwargs)
    frame_figure_6.image_rgba(image='map_rec', source=frame_source, **kwargs)

    # === Layout ===

    btn_refresh = Button(label='Refresh', width=100)
    btn_refresh.on_click(lambda _: refresh())

    btn_delete = Button(label='Delete run', width=100)
    btn_delete.on_click(lambda _: delete_run_callback())

    btn_play = Toggle(label='Play', width=100)
    btn_play.on_click(lambda on: doc.add_timeout_callback(start_play, PLAY_DELAY) if on else stop_play())
    play_callback = None

    def start_play():
        nonlocal play_callback
        play_callback = doc.add_periodic_callback(lambda: play_frame(), PLAY_INTERVAL)

    def stop_play():
        doc.remove_periodic_callback(play_callback)

    tabs = Tabs(active=0, tabs=[
                Panel(title="Metrics", child=layout([
                    [keys_table, metrics_figure],
                ])),
                Panel(title="Artifacts", child=layout([
                    [
                        layouts.column([artifacts_dir_table, artifacts_table]),
                        artifact_steps_table,
                        layout([
                            [frame_figure_1, frame_figure_2, frame_figure_3],
                            [frame_figure_4, frame_figure_5, frame_figure_6],
                        ])
                    ],
                ])),
                ])

    doc.add_root(
        layout([
            [runs_table, layouts.column([btn_refresh, btn_delete, btn_play])],
            [tabs],
        ])
    )


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
