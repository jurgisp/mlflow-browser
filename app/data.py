import pandas as pd
from bokeh.models import ColumnDataSource
import mlflow
from mlflow.tracking import MlflowClient

from .tools import *

MAX_RUNS = 100
DEFAULT_METRIC = '_loss'
TZ_LOCAL = 'Europe/Vilnius'


mlflow_client = MlflowClient()


class DataExperiments:
    def __init__(self, callback):
        self._callback = callback
        # State
        self.is_loaded = False
        self.selected_experiment_ids = []

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
        self.selected_experiment_ids = cols.get('id', [])
        print('Selected experiments: ', self.selected_experiment_ids)
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


class DataRuns:
    def __init__(self, callback, data_experiments: DataExperiments):
        self._callback = callback
        self._data_experiments = data_experiments
        # State
        self.experiment_ids = []
        self.selected_run_ids = []
        self.selected_run_df = []

        self.data = pd.DataFrame()
        self.source = ColumnDataSource(data=pd.DataFrame())
        self.source.selected.on_change('indices', lambda attr, old, new: self.on_select())  # type: ignore

    def update(self):
        experiment_ids = self._data_experiments.selected_experiment_ids
        if experiment_ids != self.experiment_ids:
            self.experiment_ids = experiment_ids
            self.source.data = self.data = self._load_data()  # type: ignore
            self.source.selected.indices = []  # type: ignore

    def on_select(self):
        cols = selected_columns(self.source)
        self.selected_run_ids = cols.get('id', [])
        self.selected_run_df = pd.DataFrame(cols)
        self._callback('runs.select')

    def _load_data(self):
        with Timer(f'mlflow.search_runs({self.experiment_ids})', verbose=True):
            df = mlflow.search_runs(self.experiment_ids, max_results=MAX_RUNS)
        if len(df) == 0:
            return df
        df['id'] = df['run_id']
        df['name'] = df['tags.mlflow.runName']
        df['start_time_local'] = dt_tolocal(df['start_time'])
        return df


def dt_tolocal(col) -> pd.Series:
    return (
        # pd.to_datetime(col, unit='s')
        col
        # .dt.tz_localize('UTC')
        .dt.tz_convert(TZ_LOCAL)
        .dt.tz_localize(None)
    )
