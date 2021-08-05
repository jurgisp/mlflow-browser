from typing import Tuple
import pandas as pd
from bokeh.models import ColumnDataSource
import mlflow
from mlflow.tracking import MlflowClient

from .tools import *

MAX_RUNS = 100
DEFAULT_METRIC = '_loss'
TZ_LOCAL = 'Europe/Vilnius'

mlflow_client = MlflowClient()


class DataAbstract:
    def __init__(self, callback, name):
        self._callback = callback
        self._name = name
        self._in_state = None
        self.data = pd.DataFrame()
        self.source = ColumnDataSource(data=pd.DataFrame())
        self.source.selected.on_change('indices', lambda attr, old, new: self.on_select())  # type: ignore

    def update(self):
        new_in_state = self.get_in_state()
        if new_in_state != self._in_state:
            self._in_state = new_in_state
            self.data = self.load_data(*self._in_state)
            self.source.data = self.data  # type: ignore
            self.source.selected.indices = []  # type: ignore

    def get_in_state(self) -> Tuple:
        return tuple()

    def on_select(self) -> None:
        self._callback(f'{self._name}.select')

    def load_data(self, state: Tuple) -> pd.DataFrame:
        raise NotImplementedError


class DataExperiments(DataAbstract):
    def __init__(self, callback, name='experiments'):
        super().__init__(callback, name)
        self.selected_experiment_ids = []

    def load_data(self):
        with Timer(f'mlflow.list_experiments()', verbose=True):
            experiments = mlflow_client.list_experiments()
        df = pd.DataFrame([
            {'id': int(e.experiment_id), 'name': e.name}
            for e in experiments
        ])
        df = df.sort_values('id', ascending=False)
        return df

    def on_select(self):
        cols = selected_columns(self.source)
        self.selected_experiment_ids = cols.get('id', [])
        super().on_select()


class DataRuns(DataAbstract):
    def __init__(self, callback, data_experiments: DataExperiments, name='runs'):
        super().__init__(callback, name)
        self._data_experiments = data_experiments
        self.selected_run_ids = []
        self.selected_run_df = pd.DataFrame()

    def get_in_state(self):
        return (self._data_experiments.selected_experiment_ids,)

    def load_data(self, experiment_ids):
        with Timer(f'mlflow.search_runs({experiment_ids})', verbose=True):
            df = mlflow.search_runs(experiment_ids, max_results=MAX_RUNS)
        if len(df) == 0:
            return df
        df['id'] = df['run_id']
        df['name'] = df['tags.mlflow.runName']
        df['start_time_local'] = dt_tolocal(df['start_time'])
        return df

    def on_select(self):
        cols = selected_columns(self.source)
        self.selected_run_ids = cols.get('id', [])
        self.selected_run_df = pd.DataFrame(cols)
        super().on_select()


def dt_tolocal(col) -> pd.Series:
    return (
        # pd.to_datetime(col, unit='s')
        col
        # .dt.tz_localize('UTC')
        .dt.tz_convert(TZ_LOCAL)
        .dt.tz_localize(None)
    )
