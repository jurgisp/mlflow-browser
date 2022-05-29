# pyright: reportGeneralTypeIssues=false
import os
from datetime import datetime
from typing import List, Optional, Tuple

import mlflow
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.models.callbacks import CustomJS
from bokeh.palettes import Category10_10, Greys3
from mlflow.tracking.client import MlflowClient

from app.mlflow_client import MlflowClientLogging

from .tools import *

pd.options.display.width = 0


RUNNING_MAX_AGE = 5 * 60  # mark as running if age is smaller than this
FAILED_DURATION = 60 * 60  # mark as failed if shorter than this

DEFAULT_METRICS = [s.strip() for s in os.environ.get('DEFAULT_METRICS', '').split(',') if s != '']  # ['agent/return', 'return', 'train_return']
DEFAULT_FILTER = os.environ.get('DEFAULT_FILTER', '')  # 'return, logprob, entropy, _loss'
DEFAULT_EXPERIMENT_IDS = [int(s) for s in os.environ.get('DEFAULT_EXPERIMENT_IDS', '').split(',') if s != '']  # [21, 22]

BASELINES_CSV = Path(__file__).parent / '../data/baselines.csv'
METRICS_CACHE_CSV = Path(__file__).parent / '../.cache/metrics.csv'

INF = 1e20  # Ignore higher metric values as infinity

TZ_LOCAL = 'Europe/Vilnius'
PALETTE = Category10_10
PALETTE_BASE = Greys3[:2]


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
    def __init__(self, callback, mlflow: MlflowClient, name='experiments'):
        super().__init__(callback, name)
        self._mlflow = mlflow

    def load_data(self):
        experiments = self._mlflow.list_experiments()
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
    def __init__(self, callback, data_experiments: DataExperiments, datac_filter: DataControl, mlflow: MlflowClientLogging, name='runs'):
        self._data_experiments = data_experiments
        self._datac_filter = datac_filter
        self._mlflow = mlflow
        super().__init__(callback, name)

    def get_in_state(self):
        return (self._data_experiments.selected_experiment_ids, self._datac_filter.value)

    def load_data(self, experiment_ids, filter):
        experiment_ids = [str(x) for x in experiment_ids or DEFAULT_EXPERIMENT_IDS]
        df: pd.DataFrame = self._mlflow.search_runs(experiment_ids)

        if len(df) == 0:
            return df
        df['id'] = df['run_id']
        df['name'] = df['tags.mlflow.runName']
        df['start_time_local'] = dt_tolocal(df['start_time'])

        # filter
        if filter:
            df = df[df['name'].str.contains(filter).fillna(False)]

        # Experiment name
        df['experiment_id'] = df['experiment_id'].astype(int)  # type: ignore
        df_exp = self._data_experiments.data.rename(columns={
            'name': 'experiment_name',
            'id': 'experiment_id'})
        df = pd.merge(df, df_exp, how='left', on='experiment_id')

        def combine_columns(df, col_names):
            df['_nan'] = np.nan
            res = df['_nan']
            for col in col_names:
                if col in df:
                    res = res.combine_first(df[col])
            return res

        # Hacky unified metrics

        df['agent_steps'] = combine_columns(df, [
            'metrics.agent_timesteps_total',  # Ray
            'metrics.time/total_timesteps',  # stable_baselines
            'metrics.train/data_steps',
            'metrics.data/steps',
            'metrics.agent/steps',
            'metrics.replay/replay_step',
            'metrics.train_replay_steps',
            'metrics._step',
            'metrics.step',
        ])
        df['grad_steps'] = combine_columns(df, [
            'metrics.training_iteration',  # Ray
            'metrics.time/iterations',  # stable_baselines
            'metrics.train_steps',
            'metrics.grad_steps',
            'metrics.train/model_grad_steps',
            'metrics._step',
            'metrics.step',
        ])
        df['return'] = combine_columns(df, [
            'metrics.episode_reward_mean',  # Ray
            'metrics.rollout/ep_rew_mean',  # stable_baselines
            'metrics.episode_reward',
            'metrics.agent_eval/return_cum100',
            'metrics.agent/return_cum100',
            'metrics.agent_eval/return_cum',
            'metrics.agent/return_cum',
            'metrics.agent_eval/return',
            'metrics.agent/return',
            'metrics.train_return',
            'metrics.return'
        ])
        df['episode_length'] = combine_columns(df, [
            'metrics.episode_len_mean',  # Ray
            'metrics.agent/episode_length',
            'metrics.actor0/length',
            'metrics.train_length',
        ])

        df['action_repeat'] = combine_columns(df, [
            'params.env_action_repeat',
            'params.env.repeat',
        ]).astype(float).fillna(1.0)
        
        df['_env_steps'] = df['agent_steps'] * df['action_repeat']
        df['env_steps'] = combine_columns(df, [
            'metrics.env_steps',
            # 'metrics.actor0/env_steps',
            # 'metrics.actor1/env_steps',
            # 'metrics.actor2/env_steps',
            '_env_steps',
        ])
        df['env_steps_ratio'] = df['env_steps'] / df['grad_steps']

        # Env/task 
        df['env'] = combine_columns(df, [
            'params.env_id',
        ])

        # Age/Duration metrics
        df['timestamp'] = combine_columns(df, [
            'metrics._timestamp',
            'metrics.timestamp'
        ])
        df['age_seconds'] = (datetime.now().timestamp() - df['timestamp'])
        df['duration_seconds'] = df['timestamp'] - df['start_time'].view(int) / 1e9
        df['fps'] = df['env_steps'] / df['duration_seconds']
        df['gps'] = df['grad_steps'] / df['duration_seconds']
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
                metrics_key = col.split('.', 1)[1]
                if not filters or any([f in metrics_key for f in filters]):
                    vals = runs_df[col].to_list()
                    if not all([v is None or np.isnan(v) for v in vals]):
                        data.append({
                            'metric': metrics_key,
                            'metric_prefix': '/'.join(metrics_key.replace('.', '/').split('/')[:-1]),  # train/loss => train
                            'metric_suffix': metrics_key.replace('.', '/').split('/')[-1],  # train/loss => loss
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


class DataRunParameters(DataAbstract):
    def __init__(self, callback, data_runs: DataRuns, datac_filter: DataControl, name='run_parameters'):
        self._data_runs = data_runs
        self._datac_filter = datac_filter
        super().__init__(callback, name)

    def get_in_state(self):
        return (self._data_runs.selected_run_ids, self._datac_filter.value)

    def load_data(self, run_ids, filter):
        runs_df = self._data_runs.selected_run_df
        if runs_df is None or len(runs_df) == 0:
            return pd.DataFrame()
        data = []
        filters = [f.strip() for f in filter.split(',') if f.strip() != ''] if filter else []  # If "filter1, filter2" allow any one of matches
        for col in sorted(runs_df.columns):
            if col.startswith('params.'):
                param_key = col.split('.', 1)[1]
                if not filters or any([f in param_key for f in filters]):
                    vals = runs_df[col].to_list()
                    if any(vals):
                        data.append({
                            'param': param_key,
                            'value1': vals[0],
                            'value2': vals[1] if len(vals) >= 2 else '',
                            'diff_color': 'red' if len(vals) >= 2 and vals[0] != vals[1] else 'black'
                        })
        return pd.DataFrame(data)


class DataMetrics(DataAbstract):
    def __init__(self,
                 callback,
                 callback_update,
                 data_runs: DataRuns,
                 data_keys: DataMetricKeys,
                 datac_smoothing: DataControl,
                 datac_envsteps: DataControl,
                 mlflow: MlflowClient,
                 name='metrics'):
        self.data_runs = data_runs
        self.data_keys = data_keys
        self.datac_smoothing = datac_smoothing
        self.datac_envsteps = datac_envsteps
        self._mlflow = mlflow
        super().__init__(callback, name, callback_update)

    def get_in_state(self):
        return (
            self.data_runs.selected_run_ids,
            self.data_keys.selected_keys or DEFAULT_METRICS,
            ['*'],
            self.datac_smoothing.value,
            self.datac_envsteps.value,
        )

    def load_data(self,
                  run_ids: List[str], 
                  metrics: List[str], 
                  baselines: List[str], 
                  smoothing_n: int, 
                  use_envsteps: int):
        runs = self.data_runs.selected_run_df
        if len(runs) > 0:
            runs = runs.sort_values(['name', 'start_time'])
        metrics = sorted(metrics)

        # Metrics

        data = []
        legends_used = set()
        i_series = 0

        for _, run in runs.iterrows():
            data_run = []
            run_id = run['id']
            run_name = run['name']
            run_env = run['env']

            envsteps_x, envsteps_y = None, None
            if use_envsteps:
                envsteps_column = None
                for col in ['train/data_steps', 'replay/replay_step', 'env_steps']:
                    if f'metrics.{col}' in run and not pd.isna(run[f'metrics.{col}']):
                        envsteps_column = col
                if envsteps_column:
                    hist = self._mlflow.get_metric_history(str(run_id), envsteps_column)
                    if len(hist) > 0:
                        hist.sort(key=lambda m: m.step)
                        envsteps_x = np.array([m.step for m in hist])
                        envsteps_y = np.array([m.value for m in hist]) * run['action_repeat']

            for metric in metrics:
                hist = self._mlflow.get_metric_history(str(run_id), metric)
                if len(hist) > 0:
                    hist.sort(key=lambda m: (m.step, m.timestamp))
                    xs = np.array([m.step for m in hist])
                    ts = np.array([m.timestamp for m in hist]) / 1000 / 3600  # Measure in hours
                    ys = np.array([m.value for m in hist])
                    ys[np.greater(np.abs(ys), INF)] = np.nan

                    if use_envsteps:
                        assert envsteps_x is not None and envsteps_y is not None
                        if len(envsteps_x) > 0:
                            # Lookup envsteps_y value for each x value
                            exs = []
                            for x in xs:  # This may be slow
                                ix = np.where(x >= envsteps_x)[0]
                                ix = ix[-1] if len(ix) > 0 else -1
                                ex = envsteps_y[ix] if ix >= 0 else 0
                                exs.append(ex)
                            xs = np.array(exs)
                        else:
                            print('WARN: no explicit env_steps, fallback to steps')

                    if smoothing_n == 1:
                        xs, ts, ys = self._apply_smoothing_samex(xs, ts, ys)
                    elif smoothing_n > 0:
                        xs, ts, ys = self._apply_smoothing_bin(xs, ts, ys, smoothing_n)

                    if len(xs) == 0:
                        continue

                    legend = f'{metric} [{run_name}]'
                    if legend in legends_used:
                        legend = f'{metric} [{run_name}] ({i_series})'
                    legends_used.add(legend)

                    range_min, range_max = self._calc_y_range(ys)
                    range_min_log, range_max_log = self._calc_y_range_log(ys)
                    data_run.append({
                        'run': run_name,
                        'env': run_env,
                        'metric': metric,
                        'legend': legend,
                        'color': PALETTE[i_series % len(PALETTE)],
                        'line_dash': 'solid',
                        'steps': xs,
                        'time': ts,
                        'values': ys,  # Note: this is np.array, if we convert to list, then we can't use nans
                        'range_min': range_min,
                        'range_max': range_max,
                        'range_min_log': range_min_log,
                        'range_max_log': range_max_log,
                        'steps_max': max(xs),
                        'time_min': min(ts),
                        'time_max': max(ts),
                    })
                    i_series += 1

            # Offset time origin

            if len(data_run) > 0:
                time_min = min(d['time_min'] for d in data_run)
                for d in data_run:
                    d['time'] -= time_min
                    d['time_min'] -= time_min
                    d['time_max'] -= time_min

            data.extend(data_run)

        # Baselines

        if baselines and len(data) > 0 and 'return' in metrics:
            envs = runs['env'].unique().tolist()
            dfb = get_baselines(baselines, envs)
            steps_max = max(d['steps_max'] for d in data)
            time_max = max(d['time_max'] for d in data)
            for i, row in dfb.iterrows():
                metric = str(row['baseline'])
                run_env = str(row['env'])
                run_name = run_env.split('-', maxsplit=1)[-1]  # Drop DMM- prefix
                val = float(row['return'])
                data.append({
                    'run': run_name,
                    'env': run_env,
                    'metric': metric,
                    'legend': f'{metric} [{run_name}]',
                    'color': PALETTE_BASE[i % len(PALETTE_BASE)],
                    'line_dash': 'dashed',
                    'steps': np.array([0, steps_max]),
                    'time': np.array([0, time_max]),
                    'values': np.array([val, val]),
                    'range_min': val,
                    'range_max': val,
                    'range_min_log': val,
                    'range_max_log': val,
                    'steps_max': steps_max,
                    'time_max': time_max,
                })

        df = pd.DataFrame(data)
        # print(df)

        # Save CSV for plot/main.py
        if len(df) > 0 and METRICS_CACHE_CSV:
            dfcsv = df.copy()
            # convert to list, because np.array doesn't serialize to CSV well
            dfcsv['steps'] = dfcsv['steps'].apply(lambda x: x.tolist())
            dfcsv['time'] = dfcsv['time'].apply(lambda x: x.tolist())
            dfcsv['values'] = dfcsv['values'].apply(lambda x: x.tolist())
            dfcsv.to_csv(str(METRICS_CACHE_CSV), index=False)
        
        return df

    def set_selected(self):
        pass

    def _apply_smoothing_bin(self, xs, ts, ys, bin_size=10):
        # Drop last partial bin
        n = (len(xs) // bin_size) * bin_size
        # For each bin: last(xs), mean(ys), last(ts)
        xs = xs[:n].reshape(-1, bin_size)[:, -1]
        ts = ts[:n].reshape(-1, bin_size)[:, -1]
        ys = ys[:n].reshape(-1, bin_size).mean(axis=1)
        return xs, ts, ys

    def _apply_smoothing_samex(self, xs, ts, ys):
        df = pd.DataFrame(dict(xs=xs, ts=ts, ys=ys))
        df = df.groupby('xs').agg({'ts': 'last', 'ys': 'mean'}).reset_index()
        return (
            df['xs'].to_numpy(),
            df['ts'].to_numpy(),
            df['ys'].to_numpy())

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
    def __init__(self, callback, data_runs: DataRuns, datac_tabs: DataControl, data_parent, is_dir: bool, mlflow: MlflowClient, name='artifacts'):
        self._data_runs = data_runs
        self._datac_tabs = datac_tabs
        self._data_parent = data_parent
        self._is_dir = is_dir
        self._mlflow = mlflow
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

        df = self._load_artifacts(run_id, parent_dir, self._is_dir)
        return df

    def _load_artifacts(self, run_id, path, dirs):
        print(f'artifact_uri: {self._data_runs.selected_run_df.iloc[0]["artifact_uri"]}')
        artifacts = self._mlflow.list_artifacts(run_id, path)
        artifacts = list([f for f in artifacts if f.is_dir == dirs])  # Filter dirs or files
        if not dirs:
            artifacts = list(reversed(artifacts))  # Order newest-first
        df = pd.DataFrame({
            'path': [f.path for f in artifacts],
            'name': [f.path.split('/')[-1] for f in artifacts],
            'file_size_mb': [f.file_size / 1024 / 1024 if f.file_size is not None else None for f in artifacts],
            'is_dir': [f.is_dir for f in artifacts],
        })
        if dirs and len(df) > 0:
            # HACK: show /0 and /1 agent episodes
            epmask = df['name'].str.startswith('episodes')
            dff = df[epmask].copy()
            df.loc[epmask, 'name'] = df.loc[epmask, 'name'] + '/0'  # type: ignore
            df.loc[epmask, 'path'] = df.loc[epmask, 'path'] + '/0'  # type: ignore
            dff['name'] = dff['name'] + '/1'  # type: ignore
            dff['path'] = dff['path'] + '/1'  # type: ignore
            df = pd.concat([df, dff])
        return df

    def set_selected(self):
        cols = selected_columns(self.source)
        self.selected_paths = cols.get('path', [])

    def reselect(self, is_refresh):
        self.source.selected.indices = []  # type: ignore


def get_baselines(baselines: List[str], envs: List[str]) -> pd.DataFrame:
    df = pd.read_csv(str(BASELINES_CSV))
    if baselines != ['*']:
        df = df[df['baseline'].isin(baselines)]
    df = df[df['env'].isin(envs)]
    return df.reset_index()
