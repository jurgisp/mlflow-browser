# %%
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import *
from bokeh import layouts
from bokeh.palettes import Category10_10 as palette

import tools

N_LINES = 10

# mlflow.set_tracking_uri('http://mlflow.threethirds.ai')
# mlflow.set_experiment('dreamer2')
mlflow_client = MlflowClient()

# %%
# runs = load_runs()
# runs.iloc[0].to_dict()
# hist = mlflow_client.get_metric_history('e27a67e35ae5434d82f7de968d984849', '_loss')
# tools.metrics_to_df(hist)

# %%


def load_runs():
    print('Loading runs...')
    df = mlflow.search_runs()
    print(f'{len(df)} runs loaded')
    return df


def load_run_metrics(run_id=None, metric='_loss'):
    if run_id is None:
        tools.metrics_to_df([])
    hist = mlflow_client.get_metric_history(run_id, metric)
    return tools.metrics_to_df(hist)


# %%
def create_app(doc):

    # Users table

    runs_source = ColumnDataSource(data=load_runs())
    runs_table = DataTable(
        source=runs_source,
        columns=[TableColumn(field="tags.mlflow.runName", title="run"),
                 TableColumn(field="start_time", title="time", formatter=DateFormatter(format="%Y-%m-%d %H:%M:%S")),
                 TableColumn(field="metrics._step", title="steps", formatter=NumberFormatter(format="0,0")),
                 TableColumn(field="metrics._loss", title="loss", formatter=NumberFormatter(format="0.00")),
                 ],
        width=1200,
        height=250,
        selectable=True
    )

    runs_source.selected.on_change('indices', lambda attr, old, new: run_selected(runs_source, new))  # pylint: disable=no-member

    def run_selected(src, ix):
        ix = ix or []
        run_ids = [src.data['run_id'][i] for i in ix]
        for i in range(N_LINES):
            run_id = run_ids[i] if i < len(run_ids) else None
            metrics_sources[i].data = load_run_metrics(run_id)

    # Metrics figure

    metrics_figure = figure(
        x_axis_label='step',
        y_axis_label='value',
        plot_width=1200,
        plot_height=600,
        tooltips=[
            ("metric", "@metric"),
            ("step", "@step"),
            ("value", "@value"),
        ]
    )
    metrics_figure.toolbar.active_scroll = metrics_figure.select_one(WheelZoomTool)

    metrics_sources = []
    for i in range(N_LINES):
        metrics_sources.append(ColumnDataSource(data=load_run_metrics()))
        metrics_figure.line(
            x='step',
            y='value',
            source=metrics_sources[i],
            color=palette[i],
            line_width=2,
            line_alpha=0.8)

    # Layout

    doc.add_root(
        layouts.column(
            runs_table,
            metrics_figure
        ))


create_app(curdoc())
