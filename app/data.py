from typing import Tuple
from bokeh.models.callbacks import CustomJS
import pandas as pd
from bokeh.models import ColumnDataSource
import mlflow
from mlflow.tracking import MlflowClient
from bokeh.palettes import Category10_10

from .tools import *


MAX_RUNS = 100
DEFAULT_METRIC = '_loss'
TZ_LOCAL = 'Europe/Vilnius'
PALETTE = Category10_10

mlflow_client = MlflowClient()


def dt_tolocal(col) -> pd.Series:
    return (
        # pd.to_datetime(col, unit='s')
        col
        # .dt.tz_localize('UTC')
        .dt.tz_convert(TZ_LOCAL)
        .dt.tz_localize(None)
    )


class DataControl:
    def __init__(self, callback, name, value=None):
        self._callback = callback
        self._name = name
        self.value = value

    def set(self, value):
        self.value = value
        self._callback(self._name)


class DataAbstract:
    def __init__(self, callback, name, callback_update=None):
        self._callback = callback
        self._callback_update = callback_update
        self._name = name
        self._in_state = None
        self.data = pd.DataFrame()
        self.source = ColumnDataSource(data=pd.DataFrame())
        self.source.selected.on_change('indices', lambda attr, old, new: self.on_select())  # type: ignore
        self.set_selected()

    def update(self, refresh=False):
        new_in_state = self.get_in_state()
        if new_in_state != self._in_state or refresh:
            self._in_state = new_in_state
            self.data = self.load_data(*self._in_state)
            self.source.data = self.data  # type: ignore
            self.reselect()
            self.set_selected()  # Update selected state immediately, but without causing additional callbacks
            if self._callback_update:
                self._callback_update(self._name)

    def on_select(self) -> None:
        self.set_selected()
        self._callback(self._name)

    def get_in_state(self) -> Tuple:  # Override
        return tuple()

    def load_data(self, state: Tuple) -> pd.DataFrame:  # Override
        raise NotImplementedError

    def set_selected(self) -> None:  # Override
        pass

    def reselect(self):  # Override (optional)
        self.source.selected.indices = []  # type: ignore


class DataExperiments(DataAbstract):
    def __init__(self, callback, name='experiments'):
        super().__init__(callback, name)

    def load_data(self):
        with Timer(f'mlflow.list_experiments()', verbose=True):
            experiments = mlflow_client.list_experiments()
        df = pd.DataFrame([
            {'id': int(e.experiment_id), 'name': e.name}
            for e in experiments
        ])
        df = df.sort_values('id', ascending=False)
        return df

    def set_selected(self):
        cols = selected_columns(self.source)
        self.selected_experiment_ids = cols.get('id', [])


class DataRuns(DataAbstract):
    def __init__(self, callback, data_experiments: DataExperiments, name='runs'):
        self._data_experiments = data_experiments
        super().__init__(callback, name)

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

    def set_selected(self):
        cols = selected_columns(self.source)
        self.selected_run_ids = cols.get('id', [])
        self.selected_run_df = pd.DataFrame(cols)

    def reselect(self):
        df = self.data
        df = df[df['id'].isin(self.selected_run_ids)]
        self.source.selected.indices = df.index.to_list()  # type: ignore


class DataMetricKeys(DataAbstract):
    def __init__(self, callback, data_runs: DataRuns, datac_filter: DataControl, name='metric_keys'):
        self._data_runs = data_runs
        self._datac_filter = datac_filter
        super().__init__(callback, name)

    def get_in_state(self):
        return (self._data_runs.selected_run_ids, self._datac_filter.value)

    def load_data(self, run_ids, filter):
        runs_df = self._data_runs.selected_run_df
        if runs_df is None or len(runs_df) == 0:
            return pd.DataFrame({'metric': [], 'value': []})
        data = []
        for col in sorted(runs_df.columns):
            if col.startswith('metrics.'):
                metrics_key = col.split('.')[1]
                if not filter or filter in metrics_key:
                    vals = runs_df[col].to_list()
                    if not all([v is None or np.isnan(v) for v in vals]):
                        data.append({
                            'metric': metrics_key,
                            'metric_prefix': '/'.join(metrics_key.split('/')[:-1]),  # train/loss => train
                            'metric_suffix': metrics_key.split('/')[-1],  # train/loss => loss
                            'value1': vals[0],
                            'value2': vals[1] if len(vals) >= 2 else np.nan,
                        })
        return pd.DataFrame(data)

    def set_selected(self):
        cols = selected_columns(self.source)
        self.selected_keys = cols.get('metric', [])

    def reselect(self):
        # Select the same metric keys, even if they are at different index
        df = self.data
        if len(df) == 0:
            self.source.selected.indices = []  # type: ignore
        else:
            df = df[df['metric'].isin(self.selected_keys)]
            self.source.selected.indices = df.index.to_list()  # type: ignore


class DataMetrics(DataAbstract):
    def __init__(self, callback, callback_update, data_runs: DataRuns, data_keys: DataMetricKeys, name='metrics'):
        self._data_runs = data_runs
        self._data_keys = data_keys
        super().__init__(callback, name, callback_update)

    def get_in_state(self):
        return (
            self._data_runs.selected_run_ids,
            self._data_keys.selected_keys or [DEFAULT_METRIC]
        )

    def load_data(self, run_ids, metrics):
        runs = self._data_runs.selected_run_df
        smoothing_n = None  # TODO
        data = []
        i = 0
        for _, run in runs.iterrows():
            run_id, run_name = run['id'], run['name']
            for metric in metrics:
                with Timer(f'mlflow.get_metric_history({metric})', verbose=True):
                    try:
                        hist = mlflow_client.get_metric_history(run_id, metric)
                    except Exception as e:
                        hist = []
                        print(f'ERROR fetching mlflow: {e}')
                if len(hist) > 0:
                    hist.sort(key=lambda m: m.timestamp)
                    xs = np.array([m.step for m in hist])
                    ts = (np.array([m.timestamp for m in hist]) - hist[0].timestamp) / 1000  # Measure in seconds
                    ys = np.array([m.value for m in hist])
                    if smoothing_n:
                        xs, ts, ys = self._apply_smoothing(xs, ts, ys, smoothing_n)
                    range_min, range_max = self._calc_y_range(ys)
                    range_min_log, range_max_log = self._calc_y_range_log(ys)
                    data.append({
                        'run': run_name,
                        'metric': metric,
                        'legend': f'{metric} [{run_name}] ({i})' if len(runs) > 1 else f'{metric} [{run_name}]',
                        'color': PALETTE[i % len(PALETTE)],
                        'steps': xs,
                        'time': ts,
                        'values': ys,
                        'range_min': range_min,
                        'range_max': range_max,
                        'range_min_log': range_min_log,
                        'range_max_log': range_max_log,
                        'steps_max': max(xs),
                        'time_max': max(ts),
                    })
                i += 1
        return pd.DataFrame(data)

    def set_selected(self):
        pass

    def _apply_smoothing(self, xs, ts, ys, bin_size=10):
        # Drop last partial bin
        n = (len(xs) // bin_size) * bin_size
        # For each bin: last(xs), mean(ys), last(ts)
        xs = xs[:n].reshape(-1, bin_size)[:, -1]
        ts = ts[:n].reshape(-1, bin_size)[:, -1]
        ys = ys[:n].reshape(-1, bin_size).mean(axis=1)
        return xs, ts, ys

    def _calc_y_range(self, ys, margin=0.05, include_zero=True):
        ys = ys[np.isfinite(ys)]
        range_min = min(0, min(ys)) if include_zero else min(ys)
        range_max = max(ys)
        dr = range_max - range_min
        if dr == 0:
            dr = 1.0
        range_min -= dr * margin
        range_max += dr * margin
        return range_min, range_max

    def _calc_y_range_log(self, ys, min_val=1e-4, margin=1.05):
        ys = ys[np.isfinite(ys)]
        range_min = max(min(ys), min_val)
        range_max = max(max(ys), min_val * margin)
        range_min /= margin
        range_max *= margin
        return range_min, range_max
