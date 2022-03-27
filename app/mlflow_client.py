from typing import List

import mlflow
from mlflow.entities.metric import Metric
from mlflow.tracking import MlflowClient
import pandas as pd

from .tools import Timer

MAX_RUNS = 500


class MlflowClientLogging(MlflowClient):

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    def list_experiments(self):
        with Timer(f'mlflow.list_experiments()', verbose=self.verbose):
            return super().list_experiments()

    def list_artifacts(self, run_id: str, path=None):
        with Timer(f'mlflow.list_artifacts({path})', verbose=self.verbose):
            return super().list_artifacts(run_id, path)

    def get_metric_history(self, run_id: str, key: str) -> List[Metric]:
        with Timer(f'mlflow.get_metric_history({key})', verbose=self.verbose):
            try:
                return super().get_metric_history(run_id, key)
            except Exception as e:
                print(f'ERROR fetching mlflow: {e}')
                return []

    def rename_run(self, run_id: str, new_name: str):
        with Timer(f'mlflow.rename_run({run_id})', verbose=self.verbose):
            super().set_tag(run_id, "mlflow.runName", new_name)

    def delete_run(self, run_id: str):
        with Timer(f'mlflow.delete_run({run_id})', verbose=self.verbose):
            return super().delete_run(run_id)

    def search_runs(self, experiment_ids) -> pd.DataFrame:
        with Timer(f'mlflow.search_runs({experiment_ids})', verbose=self.verbose):
            # NOTE: this doesn't exist on MlflowClient for some reason
            return mlflow.search_runs(experiment_ids, max_results=MAX_RUNS)  # type: ignore


class MlflowClientLoggingCaching(MlflowClientLogging):

    def __init__(self):
        super().__init__()
        self._cache_runs = {}
        self._cache_metrics = {}

    def get_metric_history(self, run_id: str, key: str) -> List[Metric]:
        res = self._cache_metrics.get((run_id, key))
        if not res:
            res = super().get_metric_history(run_id, key)
            self._cache_metrics[(run_id, key)] = res
        return res.copy()

    def search_runs(self, experiment_ids) -> pd.DataFrame:
        res = self._cache_runs.get(tuple(experiment_ids))
        if res is None:
            res = super().search_runs(experiment_ids)
            self._cache_runs[tuple(experiment_ids)] = res
        return res.copy()  # type: ignore

    def clear_cache(self):
        self._cache_runs.clear()
        self._cache_metrics.clear()
