import pandas as pd
from bokeh.models import ColumnDataSource
from mlflow.tracking import MlflowClient

from .tools import *

mlflow_client = MlflowClient()


class DataExperiments:
    def __init__(self, callback):
        self._callback = callback
        # State
        self.is_loaded = False
        self.selected_experiments = []

        self.data = pd.DataFrame()
        self.source = ColumnDataSource(data=pd.DataFrame())
        self.source.selected.on_change('indices', lambda attr, old, new: self.on_select())  # type: ignore

    def update(self):
        is_loaded = True
        if is_loaded != self.is_loaded:
            self.is_loaded = is_loaded
            self.source.data = self.data = self._load_data()  # type: ignore
            self.source.selected.indices = []  # type: ignore

    def on_select(self):
        cols = selected_columns(self.source)
        self.selected_experiments = cols.get('id', [])
        print('Selected experiments: ', self.selected_experiments)
        self._callback('experiments.select')

    def _load_data(self):
        with Timer(f'mlflow.list_experiments()', verbose=True):
            experiments = mlflow_client.list_experiments()
        df = pd.DataFrame([{
            'id': int(e.experiment_id),
            'name': e.name
        } for e in experiments])
        df = df.sort_values('id', ascending=False)
        return df
