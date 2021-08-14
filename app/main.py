# %%
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

from .tools import *
from .data import *

import artifacts_dreamer2  # for handling app-specific artifacts
import artifacts_minigrid


PLAY_INTERVAL = 500
PLAY_DELAY = 5000

SMOOTHING_OPTS = [0, 5, 10, 20]

mlflow_client = MlflowClient()


def figure(tools='pan,tap,wheel_zoom,reset', active_scroll=True, hide_axes=False, **kwargs):
    fig = bokeh.plotting.figure(
        tools=tools,
        **kwargs,
    )
    if active_scroll:
        fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)  # type: ignore
    if hide_axes:
        fig.xaxis.visible = False
        fig.yaxis.visible = False
    if kwargs.get('y_axis_type') == 'log':
        # For some reason log axis is flipped by default
        fig.y_range.flipped = True  # type: ignore
    return fig


def delete_run(run_id):
    with Timer(f'mlflow.delete_run({run_id})', verbose=True):
        mlflow_client.delete_run(run_id)


def load_artifacts(run_id=None, path=None, dirs=False):
    if run_id is None:
        return {}
    with Timer(f'mlflow.list_artifacts({path})', verbose=True):
        artifacts = mlflow_client.list_artifacts(run_id, path)
    artifacts = list([f for f in artifacts if f.is_dir == dirs])  # Filter dirs or files
    if not dirs:
        artifacts = list(reversed(artifacts))  # Order newest-first
    return {
        'path': [f.path for f in artifacts],
        'name': [f.path.split('/')[-1] for f in artifacts],
        'file_size_mb': [f.file_size / 1024 / 1024 if f.file_size is not None else None for f in artifacts],
        'is_dir': [f.is_dir for f in artifacts],
    }


def load_artifact_steps(run_id, artifact_path):
    with Timer(f'mlflow.download_artifact({artifact_path})', verbose=True):
        if artifact_path.endswith('.npz'):
            data = download_artifact_npz(mlflow_client, run_id, artifact_path)
        else:
            print(f'Artifact extension not supported: {artifact_path}')
            return {}

    print('Artifact raw: ' + str({k: v.shape for k, v in data.items()}))

    data_parsed = {}
    # if artifact_path.startswith('d2_train_batch/'):
    #     data_parsed =  artifacts_dreamer2.parse_d2_train_batch(data)
    if artifact_path.startswith('d2_wm_predict'):
        data_parsed = artifacts_dreamer2.parse_d2_wm_predict(data)
    elif artifact_path.startswith('d2_train_episodes/') or artifact_path.startswith('d2_eval_episodes/') or artifact_path.startswith('episodes/'):
        data_parsed = artifacts_dreamer2.parse_d2_episodes(data)
    else:
        print(f'Artifact type not supported: {artifact_path}')

    print('Artifact parsed: ' + str({k: v.shape for k, v in data_parsed.items()}))

    return data_parsed


def load_frame(step_data=None,
               image_keys=['image', 'image_rec', 'image_pred', 'map', 'map_agent', 'map_rec', 'map_rec_global']
               ):
    if step_data is None:
        return {k: [] for k in image_keys}

    sd = step_data

    # map_rec_global
    if 'map_rec' in sd and 'map_agent' in sd and not np.all(sd['map_rec'] == 0):
        if len(sd['map_rec'].shape) == 2 and sd['map_rec'].shape[0] > sd['map_agent'].shape[0]:
            # map_rec is bigger than map_agent (categorical) - must be agent-centric
            map_agent = artifacts_minigrid.CAT_TO_OBJ[sd['map_agent']]
            agent_pos, agent_dir = artifacts_minigrid._get_agent_pos(map_agent)
            sd['map_rec_global'] = artifacts_minigrid._map_centric_to_global(sd['map_rec'], agent_pos, agent_dir, map_agent.shape[:2])
        if len(sd['map_rec'].shape) == 3 and sd['map_rec'].shape[-1] == 3 and 'agent_pos' in sd:
            # map_rec is RGB - must be agent centric
            sd['map_rec_global'] = artifacts_minigrid.map_centric_to_global_rgb(sd['map_rec'], sd['agent_pos'], sd['agent_dir'], sd['map_agent'].shape[:2])

    data = {}
    for k in image_keys:
        obs = sd.get(k)

        # TODO: move this logic to artifacts
        if obs is None or obs.shape == (1, 1, 1):
            obs = np.zeros((1, 1, 3))

        if len(obs.shape) == 3 and obs.shape[1] == obs.shape[2]:  # Looks transposed (C,W,W)
            obs = obs.transpose(1, 2, 0)  # (C,W,W) => (W,W,C)

        if obs.shape[-1] == 3 and len(obs.shape) == 3:  # Looks like an image (H,W,C)
            img = obs
        else:
            img = artifacts_minigrid.render_obs(obs)  # Try MiniGrid
        img = to_rgba(img)
        data[k] = [img]
    return data

# %%

# load_artifacts({'id':'db1a75611d464df08f1c7052cc8b1047'})
# data = download_artifact_npz('db1a75611d464df08f1c7052cc8b1047', 'd2_train_batch/0000951.npz')
# data['imag_image'].shape


# %%

def create_app(doc):

    # === Data sources ===

    update_counter = 0

    def on_change(source, refresh=False):
        print(f'selected: {source}')
        data_experiments.update(refresh)
        data_runs.update(refresh)
        data_keys.update(refresh)
        # smoothing = SMOOTHING_OPTS[radio_smoothing.active]  # type: ignore
        data_metrics.update(refresh)
        if source == 'runs':
            run_selected(None, None, None)

        # Loader
        nonlocal update_counter
        update_counter += 1
        data_progress.data = pd.DataFrame({'counter': update_counter}, index=[0])  # type: ignore

    def on_update(source):
        print(f'updated: {source}')

    datac_keys_filter = DataControl(on_change, 'keys_filter', '')
    datac_smoothing = DataControl(on_change, 'smoothing', 0)

    data_experiments = DataExperiments(on_change)
    data_runs = DataRuns(on_change, data_experiments)
    data_keys = DataMetricKeys(on_change, data_runs, datac_keys_filter)
    data_metrics = DataMetrics(on_change, on_update, data_runs, data_keys, datac_smoothing)

    artifacts_dir_source = ColumnDataSource(data=load_artifacts())
    artifacts_source = ColumnDataSource(data=load_artifacts())
    steps_source = ColumnDataSource(data={})
    frame_source = ColumnDataSource(data=load_frame())

    # Callbacks

    def refresh():
        on_change('refresh', refresh=True)

    def run_selected(attr, old, new):
        # artifacts
        update_artifacts_dir()
        update_artifacts()
        update_steps()
        update_frame()

    def artifact_tab_selected(attr, old, new):
        # artifacts
        update_artifacts_dir()
        update_artifacts()
        update_steps()
        update_frame()

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
        ix = steps_source.selected.indices  # type: ignore
        if len(ix) == 1:
            steps_source.selected.indices = [ix[0] + 1]  # type: ignore
        else:
            steps_source.selected.indices = [0]  # type: ignore

    # Data update

    def delete_run_callback():
        if len(data_runs.selected_run_ids) == 1:
            delete_run(data_runs.selected_run_ids[0])
            on_change('delete_run', refresh=True)

    def update_artifacts_dir():
        run_id = single_or_none(data_runs.selected_run_ids) if tabs.active == 1 else None  # Don't reload if another tab
        artifacts_dir_source.data = load_artifacts(run_id, dirs=True)

    def update_artifacts():
        run_id = single_or_none(data_runs.selected_run_ids) if tabs.active == 1 else None  # Don't reload if another tab
        dir = selected_row_single(artifacts_dir_source)
        if run_id and dir:
            artifacts_source.data = load_artifacts(run_id, dir['path'])
        else:
            artifacts_source.data = {}

    def update_steps():
        run_id = single_or_none(data_runs.selected_run_ids) if tabs.active == 1 else None  # Don't reload if another tab
        artifact = selected_row_single(artifacts_source)
        if run_id and artifact:
            data = load_artifact_steps(run_id, artifact['path'])
            steps_source.data = data
        else:
            steps_source.data = {}

    def update_frame():
        step = selected_row_single(steps_source)
        frame_source.data = load_frame(step)

    # === Layout ===

    # Experiments table

    experiments_table = DataTable(
        source=data_experiments.source,
        columns=[
            # TableColumn(field="id", width=50),
            TableColumn(field="name", width=150),
        ],
        width=350,
        height=250,
        # fit_columns=False,
        selectable=True,
    )

    # Runs table

    w = 80
    runs_table = DataTable(
        source=data_runs.source,
        columns=[
            TableColumn(field="name", title="run", width=150),
            TableColumn(field="start_time_local", title="time", formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S"), width=150),
            TableColumn(field="metrics._step", title="step", formatter=NumberFormatter(format="0,0"), width=w),
            # TableColumn(field="metrics._loss", title="_loss", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="metrics.loss_model", title="loss_model", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="metrics.eval_full/logprob_img", title="eval/img", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="metrics.eval_full/logprob_map", title="eval/map", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="metrics.eval_full/acc_map", title="eval/acc_map", formatter=NumberFormatter(format="0.000"), width=w),
            #  TableColumn(field="metrics.actor_ent", title="actor_ent", formatter=NumberFormatter(format="0.00"), width=w),
            #  TableColumn(field="metrics.train_return", title="train_return", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="metrics.grad_norm", title="grad_norm", formatter=NumberFormatter(format="0.0"), width=w),
            TableColumn(field="metrics.fps", title="fps", formatter=NumberFormatter(format="0.0"), width=40),
            TableColumn(field="experiment_id", title="exp", width=40),
            TableColumn(field="run_id", title="id", width=40),
        ],
        width=1000,
        height=250,
        fit_columns=False,
        selectable=True
    )

    # Keys table

    keys_table = DataTable(
        source=data_keys.source,
        columns=[TableColumn(field="metric_prefix", title="prefix", width=50),
                 TableColumn(field="metric_suffix", title="metric", width=120),
                 TableColumn(field="value1", title="value1", formatter=NumberFormatter(format="0,0.[000]"), width=60),
                 TableColumn(field="value2", title="value2", formatter=NumberFormatter(format="0,0.[000]"), width=60),
                 ],
        width=350,
        height=600,
        fit_columns=False,
        selectable=True,
    )

    # Metrics figure

    metrics_figures = []
    for x_axis in ['steps', 'time']:
        for y_axis_type in ['linear', 'log']:
            p = figure(
                x_axis_label=x_axis,
                y_axis_label='value',
                plot_width=1000,
                plot_height=600,
                tooltips=[
                    ("run", "@run"),
                    ("metric", "@metric"),
                    (x_axis, "$x{0,0}"),
                    ("value", "$y{0,0.[000]}"),
                ],
                y_axis_type=y_axis_type,
            )
            p.xaxis[0].formatter = NumeralTickFormatter(format='0,0')
            p.multi_line(
                xs=x_axis,
                ys='values',
                source=data_metrics.source,
                color='color',
                legend_field='legend',
                line_width=2,
                line_alpha=0.8)
            # p.legend.location = 'top_left'
            metrics_figures.append(p)

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
        columns=[
            TableColumn(field="name"),
            TableColumn(field="file_size_mb", title='size (MB)', formatter=NumberFormatter(format="0,0"))
        ],
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
            TableColumn(field="reset", formatter=fmt),
            TableColumn(field="terminal", formatter=fmt),
            # TableColumn(field="reward_rec", formatter=fmt),
            # TableColumn(field="action_pred", formatter=fmt),
            # TableColumn(field="reward_pred", formatter=fmt),
            # TableColumn(field="discount_pred", formatter=fmt),
            # TableColumn(field="value", formatter=fmt),
            # TableColumn(field="value_target", formatter=fmt),
            # TableColumn(field="entropy_prior", formatter=fmt),
            # TableColumn(field="entropy_post", formatter=fmt),
            TableColumn(field="loss_kl", formatter=fmt),
            TableColumn(field="loss_image", formatter=fmt),
            TableColumn(field="logprob_img", formatter=fmt),
            TableColumn(field="loss_map", formatter=fmt),
            TableColumn(field="acc_map", formatter=fmt),
        ],
        width=600,
        height=600,
        selectable=True
    )

    steps_figure = fig = figure(
        x_axis_label='step',
        # y_axis_label='value',
        plot_width=800,
        plot_height=300,
        # tooltips=[
        #     ("run", "@run"),
        #     ("metric", "@metric"),
        #     ("step", "$x{0,0}"),
        #     ("value", "$y"),
        # ],
    )
    fig.line(x='step', y='loss_map', source=steps_source, color=palette[0], legend_label='loss_map', nonselection_alpha=1)
    fig.line(x='step', y='loss_kl', source=steps_source, color=palette[1], legend_label='loss_kl', nonselection_alpha=1)
    fig.line(x='step', y='logprob_img', source=steps_source, color=palette[2], legend_label='logprob_img', nonselection_alpha=1)
    fig.line(x='step', y='acc_map', source=steps_source, color=palette[3], legend_label='acc_map', nonselection_alpha=1)
    fig.line(x='step', y='entropy_prior', source=steps_source, color=palette[4], legend_label='prior ent.', nonselection_alpha=1, visible=False)
    fig.line(x='step', y='entropy_post', source=steps_source, color=palette[5], legend_label='posterior ent.', nonselection_alpha=1, visible=False)
    fig.legend.click_policy = "hide"

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

    # === Loader ===

    data_progress = ColumnDataSource(data=pd.DataFrame())
    data_experiments.source.selected.js_on_change('indices', CustomJS(code="""
    console.log('Data loading started');
    document.getElementById('loader_overlay').style.display = 'initial';
    """))  # Show loader when selected
    data_progress.js_on_change('data', CustomJS(code="""
    console.log('Data loading finished');
    document.getElementById('loader_overlay').style.display = 'none';
    """))  # Hide loader when data updated
    tab_progress = DataTable(source=data_progress, columns=[TableColumn(field='counter')], width=50, height=50)  # TODO: Use something invisible

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

    radio_smoothing = RadioGroup(name='Smoothing',
                                 labels=['No smoothing'] + [str(i) for i in SMOOTHING_OPTS[1:]],
                                 active=0)
    radio_smoothing.on_change('active', lambda attr, old, new: datac_smoothing.set(SMOOTHING_OPTS[new]))  # type: ignore

    txt_metric_filter = TextInput(title="Filter:", width=350)
    txt_metric_filter.on_change('value_input', lambda attr, old, new: datac_keys_filter.set(new))  # type: ignore

    tabs = Tabs(active=0, tabs=[
                Panel(title="Metrics", child=layout([
                    [
                        layout([
                            [txt_metric_filter],
                            [keys_table],
                        ]),
                        Tabs(active=0, tabs=[
                            Panel(title="Linear/Steps", child=metrics_figures[0]),
                            Panel(title="Log/Steps", child=metrics_figures[1]),
                            Panel(title="Linear/Time", child=metrics_figures[2]),
                            Panel(title="Log/Time", child=metrics_figures[3]),
                        ]),
                        radio_smoothing,
                    ],
                ])),
                Panel(title="Artifacts", child=layout([
                    [
                        layouts.column([artifacts_dir_table, artifacts_table]),
                        artifact_steps_table,
                        layout([
                            [frame_figure_1, frame_figure_2, frame_figure_3],
                            [frame_figure_4, frame_figure_5, frame_figure_6],
                            [steps_figure],
                        ])
                    ],
                ])),
                ])
    tabs.on_change('active', artifact_tab_selected)

    doc.add_root(
        layout([
            [
                experiments_table,
                runs_table,
                layouts.column([btn_refresh, btn_delete, btn_play]),
            ],
            [tabs],
            [tab_progress],
        ])
    )

    # ----- Start -----

    on_change('init')


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
    curdoc().title = 'Mlflow'
