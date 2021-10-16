import os
from typing import Optional, Tuple
from datetime import datetime
from bokeh.models.callbacks import CustomJS
import pandas as pd
from bokeh.models import ColumnDataSource
import mlflow
from mlflow.tracking import MlflowClient
from bokeh.palettes import Category10_10

from .tools import *


MAX_RUNS = 500
RUNNING_MAX_AGE = 5 * 60  # mark as running if age is smaller than this
FAILED_DURATION = 90 * 60  # mark as failed if shorter than this
# DEFAULT_METRICS = ['_loss']
DEFAULT_METRICS = []
# DEFAULT_FILTER = 'train/, agent/'
DEFAULT_FILTER = ''

DEFAULT_EXPERIMENT_IDS = [int(s) for s in (os.environ.get('DEFAULT_EXPERIMENT_IDS') or '').split(',') if s != '']
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
    def __init__(self, callback, name, callback_update=None, loader_on_select=True):
        self._callback = callback
        self._callback_update = callback_update
        self._name = name
        self._in_state = None
        self.data = pd.DataFrame()
        self.source = ColumnDataSource(data=pd.DataFrame())
        self.source.selected.on_change('indices', lambda attr, old, new: self.on_select())  # type: ignore
        if loader_on_select:
            # Show loader when selected
            self.source.selected.js_on_change('indices', CustomJS(code="document.getElementById('loader_overlay').style.display = 'initial'"))  # type: ignore
        self.set_selected()

    def update(self, refresh=False, quick=False):
        new_in_state = self.get_in_state()
        if new_in_state != self._in_state or refresh:
            self._in_state = new_in_state
            self.data = self.load_data(*self._in_state)
            self.source.data = self.data  # type: ignore
            if not quick:
                self.reselect(refresh)
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

    def reselect(self, is_refresh):  # Override (optional)
        if not is_refresh:
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
    def __init__(self, callback, data_experiments: DataExperiments, datac_filter: DataControl, name='runs'):
        self._data_experiments = data_experiments
        self._datac_filter = datac_filter
        super().__init__(callback, name)

    def get_in_state(self):
        return (self._data_experiments.selected_experiment_ids, self._datac_filter.value)

    def load_data(self, experiment_ids, filter):
        experiment_ids = experiment_ids or DEFAULT_EXPERIMENT_IDS
        with Timer(f'mlflow.search_runs({experiment_ids})', verbose=True):
            df = mlflow.search_runs(experiment_ids, max_results=MAX_RUNS)
        if len(df) == 0:
            return df
        df['id'] = df['run_id']
        df['name'] = df['tags.mlflow.runName']
        df['start_time_local'] = dt_tolocal(df['start_time'])

        # filter
        if filter:
            df = df[df['name'].str.contains(filter)]

        # Experiment name
        df['experiment_id'] = df['experiment_id'].astype(int)  # type: ignore
        df_exp = self._data_experiments.data.rename(columns={
            'name': 'experiment_name',
            'id': 'experiment_id'})
        df = pd.merge(df, df_exp, how='left', on='experiment_id')

        def combine_columns(df, col_names):
            res = None
            for col in col_names:
                if col in df:
                    if res is None:
                        res = df[col]
                    else:
                        res = res.combine_first(df[col])
            return res if res is not None else np.nan

        # Hacky unified metrics
        df['agent_steps'] = combine_columns(df, ['metrics.train/data_steps',
                                                 'metrics.data/steps',
                                                 'metrics.agent/steps',
                                                 'metrics.train_replay_steps'])
        df['agent_steps_x4'] = df['agent_steps'] * 4
        df['return'] = combine_columns(df, ['metrics.agent_eval/return_cum100',
                                            'metrics.agent/return_cum100',
                                            'metrics.agent_eval/return',
                                            'metrics.agent/return',
                                            'metrics.train_return'])
        df['episode_length'] = combine_columns(df, ['metrics.agent/episode_length', 'metrics.train_length'])
        df['fps'] = combine_columns(df, ['metrics.train/fps', 'metrics.fps'])

        # Age/Duration metrics
        if 'metrics._timestamp' in df:
            df['age_seconds'] = (datetime.now().timestamp() - df['metrics._timestamp'])
            df['duration_seconds'] = df['metrics._timestamp'] - df['start_time'].view(int) / 1e9
        else:
            df['age_seconds'] = np.nan
            df['duration_seconds'] = np.nan
        df['age'] = df['age_seconds'].apply(lambda a: f'{int(a/60)} min' if a < 3600 else f'{int(a/3600)} h' if a < 86400 else f'{int(a/86400)} d' if a > 0 else '')
        df['duration'] = df['duration_seconds'].apply(lambda a: f'{int(a/60)} min' if a < 3600 else f'{int(a/3600)} h' if a > 0 else '')

        # Status color
        df['status_color'] = 'black'
        df.loc[df['age_seconds'] < RUNNING_MAX_AGE, 'status_color'] = 'green'
        df.loc[(df['age_seconds'] > RUNNING_MAX_AGE) & (df['duration_seconds'] < FAILED_DURATION), 'status_color'] = 'red'
        return df

    def set_selected(self):
        cols = selected_columns(self.source)
        self.selected_run_ids = cols.get('id', [])
        self.selected_run_df = pd.DataFrame(cols)

    def reselect(self, is_refresh):
        df = self.data
        if len(df) == 0:
            self.source.selected.indices = []  # type: ignore
        else:
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
        filters = [f.strip() for f in filter.split(',') if f.strip() != ''] if filter else []  # If "filter1, filter2" allow any one of matches
        for col in sorted(runs_df.columns):
            if col.startswith('metrics.'):
                metrics_key = col.split('.')[1]
                if not filters or any([f in metrics_key for f in filters]):
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

    def reselect(self, is_refresh):
        # Select the same metric keys, even if they are at different index
        df = self.data
        if len(df) == 0:
            self.source.selected.indices = []  # type: ignore
        else:
            df = df[df['metric'].isin(self.selected_keys)]
            self.source.selected.indices = df.index.to_list()  # type: ignore


class DataMetrics(DataAbstract):
    def __init__(self, callback, callback_update, data_runs: DataRuns, data_keys: DataMetricKeys, datac_smoothing: DataControl, name='metrics'):
        self._data_runs = data_runs
        self._data_keys = data_keys
        self._datac_smoothing = datac_smoothing
        super().__init__(callback, name, callback_update)

    def get_in_state(self):
        return (
            self._data_runs.selected_run_ids,
            self._data_keys.selected_keys or DEFAULT_METRICS,
            self._datac_smoothing.value
        )

    def load_data(self, run_ids, metrics, smoothing_n):
        runs = self._data_runs.selected_run_df
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
                    ts = ts / 3600  # Measure in hours
                    ys = np.array([m.value for m in hist])
                    ys[np.isposinf(ys) | np.isneginf(ys)] = np.nan
                    if smoothing_n:
                        xs, ts, ys = self._apply_smoothing(xs, ts, ys, smoothing_n)
                    if len(xs) == 0:
                        continue
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
        df = pd.DataFrame(data)
        return df

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
        if len(ys) == 0:
            return -margin, margin
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
        if len(ys) == 0:
            return min_val / margin, min_val * margin
        range_min = max(min(ys), min_val)
        range_max = max(max(ys), min_val * margin)
        range_min /= margin
        range_max *= margin
        return range_min, range_max


class DataArtifacts(DataAbstract):
    def __init__(self, callback, data_runs: DataRuns, datac_tabs: DataControl, data_parent, is_dir: bool, name='artifacts'):
        self._data_runs = data_runs
        self._datac_tabs = datac_tabs
        self._data_parent = data_parent
        self._is_dir = is_dir
        super().__init__(callback, name)

    def get_in_state(self):
        parent_dirs = self._data_parent.selected_paths if self._data_parent else None
        return (self._data_runs.selected_run_ids, parent_dirs, self._datac_tabs.value)

    def load_data(self, run_ids, parent_dirs, tab):
        if tab != 'artifacts':
            return pd.DataFrame()

        if len(run_ids) != 1:
            return pd.DataFrame()
        run_id = run_ids[0]

        if parent_dirs is not None:
            if len(parent_dirs) != 1:
                return pd.DataFrame()
            parent_dir = parent_dirs[0]
        else:
            parent_dir = None

        if parent_dir == 'episodes' or parent_dir == 'episodes_eval':
            parent_dir = parent_dir + '/0'  # HACK
        df = self._load_artifacts(run_id, parent_dir, self._is_dir)
        return df

    def _load_artifacts(self, run_id, path, dirs):
        with Timer(f'mlflow.list_artifacts({path})', verbose=True):
            artifacts = mlflow_client.list_artifacts(run_id, path)
        artifacts = list([f for f in artifacts if f.is_dir == dirs])  # Filter dirs or files
        if not dirs:
            artifacts = list(reversed(artifacts))  # Order newest-first
        return pd.DataFrame({
            'path': [f.path for f in artifacts],
            'name': [f.path.split('/')[-1] for f in artifacts],
            'file_size_mb': [f.file_size / 1024 / 1024 if f.file_size is not None else None for f in artifacts],
            'is_dir': [f.is_dir for f in artifacts],
        })

    def set_selected(self):
        cols = selected_columns(self.source)
        self.selected_paths = cols.get('path', [])

    def reselect(self, is_refresh):
        self.source.selected.indices = []  # type: ignore
