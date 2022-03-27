import ast
from pathlib import Path

import bokeh.plotting
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import *  # type: ignore
from bokeh.plotting import curdoc

METRICS_CACHE_CSV = Path(__file__).parent / '../.cache/metrics.csv'
GROUP_BY = 'env'


def create_app(doc):

    # Load data

    df: pd.DataFrame = pd.read_csv(METRICS_CACHE_CSV)  # type: ignore
    for col in ['steps', 'time', 'values']:
        df[col] = df[col].apply(ast.literal_eval)
    df['env'] = df['env'].apply(lambda s: s.split('-', maxsplit=1)[-1] if isinstance(s, str) else s)  # Drop prefix
    # df['legend'] = df['metric']

    # Subplots

    figs = []
    for key, dfg in df.groupby(GROUP_BY):
        p = figure(
            x_axis_label='steps',
            y_axis_label='value',
            tooltips=[
                ("run", "@run"),
                ("metric", "@metric"),
                ("steps", "$x{0,0}"),
                ("value", "$y{0,0.[000]}"),
            ],
            title=key
        )
        p.xaxis[0].formatter = BasicTickFormatter(precision=1, use_scientific=True)
        p.multi_line(
            xs='steps',
            ys='values',
            source=ColumnDataSource(data=dfg),
            color='color',
            # legend_field='legend',
            line_width=2,
            line_alpha=0.8,
            line_dash='line_dash')
        # p.legend.location = 'top_left'
        if figs:
            p.x_range = figs[0].x_range  # Share x axis
        figs.append(p)

    # Layout

    grid = gridplot(figs, ncols=3, plot_width=400, plot_height=300)
    doc.add_root(grid)


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
    return fig


if __name__.startswith('bokeh_app_'):
    create_app(curdoc())
    curdoc().title = 'Plot'
