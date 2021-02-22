# %%
import mlflow
import pandas as pd

from bokeh.plotting import figure, curdoc
from bokeh.models import *
from bokeh import layouts

# %%
# mlflow.set_tracking_uri('http://mlflow.threethirds.ai')
# mlflow.set_experiment('dreamer2')


# %%
def load_runs():
    print('Loading runs...')
    df = mlflow.search_runs()
    print(f'{len(df)} runs loaded')
    return df


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
        width=600,
        height=600,
        selectable=True
    )

    # Layout

    doc.add_root(
        layouts.row(
            runs_table,
        ))


create_app(curdoc())
