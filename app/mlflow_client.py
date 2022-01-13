from typing import List

from mlflow.entities.metric import Metric
from mlflow.tracking import MlflowClient

from .tools import Timer


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

    def delete_run(self, run_id: str):
        with Timer(f'mlflow.delete_run({run_id})', verbose=self.verbose):
            return super().delete_run(run_id)


class MlflowClientLoggingCaching(MlflowClientLogging):

    def __init__(self):
        super().__init__()
        self._cache = {}

    def get_metric_history(self, run_id: str, key: str) -> List[Metric]:
        res = self._cache.get((run_id, key))
        if not res:
            res = super().get_metric_history(run_id, key)
            self._cache[(run_id, key)] = res
        return res.copy()

    def clear_cache(self):
        self._cache.clear()