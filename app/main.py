import bokeh.plotting
from bokeh import layouts
from bokeh.layouts import layout
from bokeh.models import (Button, DataTable, DateFormatter,
                          HTMLTemplateFormatter, NumberFormatter,
                          NumeralTickFormatter, Panel, PreText, RadioGroup,
                          TableColumn, Tabs, TextInput, Toggle, WheelZoomTool)
from bokeh.palettes import Category10_10 as palette
from bokeh.plotting import curdoc

from .artifacts import load_artifact_steps, render_step_frames
from .data import *
from .mlflow_client import MlflowClientLoggingCaching
from .tools import *

PLAY_INTERVAL = 100
PLAY_DELAY = 0
DEFAULT_TAB = 'metrics'
TABLE_HEIGHT = 300

SMOOTHING_OPTS = [0, 1, 4, 10, 30, 100]

mlclient = MlflowClientLoggingCaching()


def figure(tools='pan,tap,wheel_zoom,reset', active_scroll=True, hide_axes=False, **kwargs):
    fig = bokeh.plotting.figure(
        tools=tools,
        **kwargs,
    )
    if active_scroll:
        fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)
    if hide_axes:
        fig.xaxis.visible = False
        fig.yaxis.visible = False
    # if kwargs.get('y_axis_type') == 'log':
    #     # For some reason log axis is flipped by default
    #     fig.y_range.flipped = True  # type: ignore
    return fig


def create_app(doc):

    # === Data sources ===

    progress_counter = 0

    def on_change(source, refresh=False):
        print(f'selected: {source}')
        if refresh:
            mlclient.clear_cache()
        data_experiments.update(refresh)
        data_runs.update(refresh)
        data_params.update(refresh)
        data_keys.update(refresh)
        data_metrics.update(refresh)
        data_artifacts_dir.update(refresh)
        data_artifacts.update(refresh)

        if source == 'artifacts':
            artifact_selected(None, None, None)

        if source == 'runs':
            # Set txt_rename to current name
            if len(data_runs.selected_run_df) == 1:
                txt_rename.value = data_runs.selected_run_df.iloc[0]['name'] or ''  # type: ignore
            else:
                txt_rename.value = ''  # type: ignore

        # Loader
        nonlocal progress_counter
        progress_counter += 1
        text_progress.text = str(progress_counter)

    def on_update(source):
        print(f'updated: {source}')

    datac_keys_filter = DataControl(on_change, 'keys_filter', DEFAULT_FILTER)
    datac_runs_filter = DataControl(on_change, 'runs_filter', '')
    datac_smoothing = DataControl(on_change, 'smoothing', 1)
    datac_envsteps = DataControl(on_change, 'envsteps', 0)
    datac_tabs = DataControl(on_change, 'tabs', DEFAULT_TAB)

    data_experiments = DataExperiments(on_change, mlclient)
    data_runs = DataRuns(on_change, data_experiments, datac_runs_filter, mlclient)
    data_params = DataRunParameters(on_change, data_runs, datac_keys_filter)
    data_keys = DataMetricKeys(on_change, data_runs, datac_keys_filter)
    data_metrics = DataMetrics(on_change, on_update, data_runs, data_keys, datac_smoothing, datac_envsteps, mlclient)
    data_artifacts_dir = DataArtifacts(on_change, data_runs, datac_tabs, None, True, mlclient, 'artifacts_dir')
    data_artifacts = DataArtifacts(on_change, data_runs, datac_tabs, data_artifacts_dir, False, mlclient, 'artifacts')

    steps_source = ColumnDataSource(data={})
    frame_source = ColumnDataSource(data=render_step_frames())
    frame_play_counter = 0

    # Callbacks

    def refresh():
        on_change('refresh', refresh=True)

    def artifact_selected(attr, old, new):
        update_steps()
        update_frame()

    def step_selected(attr, old, new):
        update_frame()
    steps_source.selected.on_change('indices', step_selected)  # type: ignore

    def play_frame():
        nonlocal frame_play_counter
        frame_play_counter += 1
        update_frame(frame_play_counter)

    # Data update

    def rename_run_callback():
        run_id = data_runs.selected_run_ids[0] if len(data_runs.selected_run_ids) == 1 else None
        new_name = str(txt_rename.value).strip()
        if run_id and new_name != '':
            mlclient.rename_run(run_id, new_name)
            on_change('rename_run', refresh=True)
        else:
            on_change('rename_run', refresh=False)

    def delete_run_callback():
        for run_id in data_runs.selected_run_ids:
            mlclient.delete_run(run_id)
        on_change('delete_run', refresh=True)

    def update_steps():
        run_id = single_or_none(data_runs.selected_run_ids) if tabs.active == 1 else None  # Don't reload if another tab
        artifact_path = single_or_none(data_artifacts.selected_paths)
        if run_id and artifact_path:
            data = load_artifact_steps(mlclient, run_id, artifact_path)
            steps_source.data = data  # type: ignore
            if len(data) > 0:
                steps_source.selected.indices = [0]  # type: ignore
        else:
            steps_source.data = {}  # type: ignore
            steps_source.selected.indices = []  # type: ignore

    def update_frame(offset=0):
        steps = selected_rows(steps_source)
        step = steps[offset % len(steps)] if len(steps) > 0 else None
        frame_source.data = render_step_frames(step)  # type: ignore

    # === Layout ===

    # Experiments table

    experiments_table = DataTable(
        source=data_experiments.source,
        columns=[
            TableColumn(field="name", width=130),
            TableColumn(field="id", width=50),
        ],
        width=200,
        height=TABLE_HEIGHT,
        fit_columns=False,
        selectable=True,
    )

    # Runs table

    w = 60
    mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    runs_table = DataTable(
        source=data_runs.source,
        columns=[
            TableColumn(field="experiment_name", title="exp", width=100),
            # TableColumn(field="params.resume_id", title="id", width=60),
            TableColumn(field="name", title="run", width=250),
            # TableColumn(field="run_id", width=40),
            # TableColumn(field="artifact_uri", title="artifact_uri", width=w),
            # TableColumn(field="params.entropy", title="ent", width=60),
            # TableColumn(field="params.kl_weight", title="kl", width=60),
            TableColumn(field="age", title="age", width=60,
                        formatter=HTMLTemplateFormatter(template="<span style='color:<%= status_color %>'><%= value %></span>")),
            TableColumn(field="duration", title="duration", width=60),
            TableColumn(field="start_time_local", title="time", formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S"), width=140),
            TableColumn(field="return", title="return", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="entropy", title="entropy", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="grad_steps", title="grad_steps", formatter=NumberFormatter(format="0,0"), width=80),
            TableColumn(field="env_steps", title="env_steps", formatter=NumberFormatter(format="0,0"), width=80),
            TableColumn(field="gps", title="gps", formatter=NumberFormatter(format="0.00"), width=w),
            TableColumn(field="fps", title="fps", formatter=NumberFormatter(format="0.00"), width=w),
            # TableColumn(field="metrics.train/visit_memsize", title="memsize", formatter=NumberFormatter(format="0"), width=w),
            # TableColumn(field="metrics.eval/logprob_image", title="logprob_image(eval)", formatter=NumberFormatter(format="0.00"), width=w),
            # TableColumn(field="metrics.train/loss_image", title="loss_image(train)", formatter=NumberFormatter(format="0.00"), width=w),
            # TableColumn(field="metrics.train/loss_kl", title="loss_kl(train)", formatter=NumberFormatter(format="0.00"), width=w),
            # TableColumn(field="action_repeat", title="action_repeat", formatter=NumberFormatter(format="0"), width=w),
            TableColumn(field="env_steps_ratio", title="step_ratio", formatter=NumberFormatter(format="0"), width=w),
            TableColumn(field="episode_length", title="ep_length", formatter=NumberFormatter(format="0"), width=w),
            # TableColumn(field="metrics.eval_full/acc_map", title="eval/acc_map", formatter=NumberFormatter(format="0.000"), width=w),
            # TableColumn(field="metrics.train/grad_norm", title="grad_norm", formatter=NumberFormatter(format="0.0"), width=w),
            TableColumn(field="run_id", title="id", width=40,
                        formatter=HTMLTemplateFormatter(template=f"<a href='{mlflow_tracking_uri}/#/experiments/<%= experiment_id %>/runs/<%= value %>' target='_blank'><%= value %></a>")),
            # TableColumn(field="env", title="env", width=100),
        ],
        width=1150,
        height=TABLE_HEIGHT,
        fit_columns=False,
        selectable=True
    )

    # Params table

    params_table = DataTable(
        source=data_params.source,
        columns=[TableColumn(field="param", width=230),
                 TableColumn(field="value1", width=90,
                             formatter=HTMLTemplateFormatter(template="<span style='color:<%= diff_color %>'><%= value %></span>")),
                 TableColumn(field="value2", width=90,
                             formatter=HTMLTemplateFormatter(template="<span style='color:<%= diff_color %>'><%= value %></span>")),
                 ],
        width=450,
        height=600,
        fit_columns=False,
        selectable=True,
    )

    # Keys table

    keys_table = DataTable(
        source=data_keys.source,
        columns=[TableColumn(field="metric_prefix", title="prefix", width=100),
                 TableColumn(field="metric_suffix", title="metric", width=170),
                 TableColumn(field="value1", title="value1", formatter=NumberFormatter(format="0,0.[000]"), width=70),
                 TableColumn(field="value2", title="value2", formatter=NumberFormatter(format="0,0.[000]"), width=70),
                 ],
        width=450,
        height=600,
        fit_columns=False,
        selectable=True,
    )

    # Metrics figure

    metrics_figures = []
    for x_axis in ['steps', 'time']:
        for y_axis_type in ['linear', 'log']:
            p = figure(
                x_axis_label='hours' if x_axis == 'time' else x_axis,
                y_axis_label='value',
                plot_width=900,
                plot_height=600,
                tooltips=[
                    ("run", "@run"),
                    ("metric", "@metric"),
                    (x_axis, "$x{0,0}" if x_axis == 'steps' else "$x{0.0}"),
                    ("value", "$y{0,0.[000]}"),
                ],
                y_axis_type=y_axis_type,
            )
            # p.yaxis[0].formatter = NumeralTickFormatter(format='00.0')
            p.xaxis[0].formatter = NumeralTickFormatter(format='0,0' if x_axis == 'steps' else '0.0')
            p.multi_line(
                xs=x_axis,
                ys='values',
                source=data_metrics.source,
                color='color',
                legend_field='legend',
                line_width=2,
                line_alpha=0.8,
                line_dash='line_dash')
            p.legend.location = 'top_left'
            metrics_figures.append(p)

    # === Artifacts ===

    # Artifacts list

    artifacts_dir_table = DataTable(
        source=data_artifacts_dir.source,
        columns=[TableColumn(field="path", title="directory")],
        width=300,
        height=250,
        selectable=True,
    )

    artifacts_table = DataTable(
        source=data_artifacts.source,
        columns=[
            TableColumn(field="name", width=200),
            TableColumn(field="file_size_mb", title='size (MB)', formatter=NumberFormatter(format="0,0"), width=50)
        ],
        width=300,
        height=500,
        fit_columns=False,
        selectable=True,
    )

    # Episode steps

    steps_figures = [
        figure(
            tools='box_zoom,box_select,wheel_zoom,tap,reset',
            x_axis_label='step',
            plot_width=900,
            plot_height=450,
        ),
        figure(
            tools='box_zoom,box_select,wheel_zoom,tap,reset',
            x_axis_label='step',
            plot_width=900,
            plot_height=450,
        ),
    ]
    for i, (ifig, metric, visible) in enumerate([
        (0, 'loss_map', 1),
        (0, 'loss_kl', 1),
        (0, 'loss_goal_direction', 1),
        (0, 'logprob_image', 0),
        (0, 'acc_map', 0),
        # ('entropy_prior', 1),
        # ('entropy_post', 1),
        (1, 'value', 1),
        (1, 'value_target', 1),
        (1, 'value_aux', 0),
        (1, 'return_discounted', 0),
        (1, 'return', 1),
        (1, 'reward', 0),
        (1, 'reward_pred', 1),
        (1, 'vecnovel', 0),
    ]):
        fig = steps_figures[ifig]
        fig.line(x='step', y=metric, source=steps_source, color=palette[i % len(palette)], legend_label=metric, nonselection_alpha=1, visible=visible == 1)
        fig.circle(x='step', y=metric, source=steps_source, color=palette[i % len(palette)], legend_label=metric, nonselection_alpha=0, visible=visible == 1)
    for fig in steps_figures:
        fig.legend.click_policy = 'hide'

    fmt = NumberFormatter(format="0.[00]")
    artifact_steps_table = DataTable(
        source=steps_source,
        columns=[
            TableColumn(field="step", formatter=NumberFormatter(format="0,0")),
            TableColumn(field="action", title='action (last)', formatter=fmt),
            TableColumn(field="reward", title='reward (last)', formatter=fmt),
            TableColumn(field="reset", formatter=fmt),
            TableColumn(field="terminal", formatter=fmt),
            # TableColumn(field="reward_rec", formatter=fmt),
            TableColumn(field="action_pred", formatter=fmt),
            TableColumn(field="reward_pred", formatter=NumberFormatter(format="0.[000]")),
            TableColumn(field="terminal_pred", formatter=fmt),
            # TableColumn(field="value_target", formatter=fmt),
            TableColumn(field="entropy_prior", formatter=fmt),
            # TableColumn(field="entropy_post", formatter=fmt),
            TableColumn(field="loss_kl", formatter=fmt),
            TableColumn(field="value", formatter=fmt),
            TableColumn(field="value_target", formatter=fmt),
            TableColumn(field="value_weight", formatter=fmt),
            TableColumn(field="value_advantage", formatter=fmt),
            # TableColumn(field="return_discounted", formatter=fmt),
            # TableColumn(field="loss_image", formatter=fmt),
            # TableColumn(field="logprob_image", formatter=fmt),
            # TableColumn(field="loss_map", formatter=fmt),
            # TableColumn(field="acc_map", formatter=fmt),
        ],
        width=900,
        height=300,
        selectable=True
    )

    # Frame

    kwargs = dict(plot_width=250, plot_height=250, x_range=[0, 10], y_range=[0, 10], toolbar_location=None, active_scroll=False, hide_axes=True)
    frame_figure_1 = fig = figure(title='Observation', **kwargs)
    frame_figure_2 = fig = figure(title='Prediction', **kwargs)
    frame_figure_3 = fig = figure(title='Reconstruction', **kwargs)
    frame_figure_4 = fig = figure(title='Environment', **kwargs)
    frame_figure_5 = fig = figure(title='Map', **kwargs)
    frame_figure_6 = fig = figure(title='Map prediction', **kwargs)
    kwargs = dict(x=0, y=0, dw=10, dh=10)
    frame_figure_1.image_rgba(image='image', source=frame_source, **kwargs)
    frame_figure_2.image_rgba(image='image_pred', source=frame_source, **kwargs)
    frame_figure_3.image_rgba(image='image_rec', source=frame_source, **kwargs)
    frame_figure_4.image_rgba(image='map_agent', source=frame_source, **kwargs)
    frame_figure_5.image_rgba(image='map', source=frame_source, **kwargs)
    frame_figure_6.image_rgba(image='map_rec', source=frame_source, **kwargs)

    # === Loader ===

    text_progress = PreText(text='', visible=False)  # This is dummy hidden text, just to trigger js_on_change events
    text_progress.js_on_change('text', CustomJS(code="document.getElementById('loader_overlay').style.display = 'none'"))  # type: ignore

    # === Layout ===

    btn_refresh = Button(label='Refresh', width=100)
    btn_refresh.on_click(lambda _: refresh())
    btn_refresh.js_on_click(CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))

    btn_rename = Button(label='Rename run', width=100)
    btn_rename.on_click(lambda _: rename_run_callback())
    btn_rename.js_on_click(CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))

    btn_delete = Button(label='Delete run', width=100)
    btn_delete.on_click(lambda _: delete_run_callback())
    btn_delete.js_on_click(CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))

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
                                 active=1)
    radio_smoothing.on_change('active', lambda attr, old, new: datac_smoothing.set(SMOOTHING_OPTS[new]))  # type: ignore
    radio_smoothing.js_on_change('active', CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))  # type: ignore

    radio_envsteps = RadioGroup(name='X axis',
                                labels=['Steps', 'Env steps'],
                                active=datac_envsteps.value)
    radio_envsteps.on_change('active', lambda attr, old, new: datac_envsteps.set(new))  # type: ignore
    radio_envsteps.js_on_change('active', CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))  # type: ignore

    txt_metric_filter = TextInput(title="Filter:", width=350, value=datac_keys_filter.value)
    txt_metric_filter.on_change('value', lambda attr, old, new: datac_keys_filter.set(new))  # type: ignore
    txt_metric_filter.js_on_change('value', CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))  # type: ignore

    txt_runs_filter = TextInput(title="Search runs:", width=200, value=datac_runs_filter.value)
    txt_runs_filter.on_change('value', lambda attr, old, new: datac_runs_filter.set(new))  # type: ignore
    txt_runs_filter.js_on_change('value', CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))  # type: ignore

    txt_rename = TextInput(width=200)

    tabs = Tabs(active=1 if datac_tabs.value == 'artifacts' else 0, tabs=[
                Panel(title="Metrics", child=layout([
                    [
                        layout([
                            [txt_metric_filter],
                            [
                                Tabs(active=0, tabs=[
                                    Panel(title="Metrics", child=keys_table),
                                    Panel(title="Parameters", child=params_table),
                                ]),
                            ],
                        ]),
                        Tabs(active=0, tabs=[
                            Panel(title="Linear/Steps", child=metrics_figures[0]),
                            # Panel(title="Log/Steps", child=metrics_figures[1]),
                            Panel(title="Linear/Time", child=metrics_figures[2]),
                            # Panel(title="Log/Time", child=metrics_figures[3]),
                        ]),
                        layout([
                            [radio_smoothing],
                            [radio_envsteps],
                        ]),
                    ],
                ])),
                Panel(title="Artifacts", child=layout([
                    [
                        layouts.column([artifacts_dir_table, artifacts_table]),  # type: ignore
                        layout([
                            [
                                layout([
                                    [artifact_steps_table],
                                    [
                                        Tabs(active=0, tabs=[
                                            Panel(title="Rewards", child=steps_figures[1]),
                                            Panel(title="Losses", child=steps_figures[0]),
                                        ])
                                    ],
                                ]),
                                layout([
                                    [frame_figure_1, frame_figure_4],
                                    [frame_figure_2, frame_figure_3],
                                    [frame_figure_6, frame_figure_5],

                                ])
                            ]
                        ]),
                    ],
                ])),
                ])
    tabs.on_change('active', lambda attr, old, new: datac_tabs.set('artifacts' if new == 1 else 'metrics'))  # type: ignore
    tabs.js_on_change('active', CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))  # type: ignore

    doc.add_root(
        layout([
            [
                experiments_table,
                runs_table,
                layouts.column([  # type: ignore
                    txt_runs_filter,
                    btn_refresh,
                    btn_delete,
                    btn_play,
                    txt_rename,
                    btn_rename,
                ]),
            ],
            [tabs],
            [text_progress],
        ])
    )

    # ----- Start -----

    def startup():
        on_change('init')

    doc.add_timeout_callback(startup, 500)


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
    curdoc().title = 'Mlflow'
