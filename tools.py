import time
import pandas as pd


def metrics_to_df(metric_history, run=None):
    return pd.DataFrame(
        [
            [run, m.key, m.timestamp, m.step, m.value]
            for m in metric_history
        ],
        columns=['run', 'metric', 'timestamp', 'step', 'value']
    )


class Timer:

    def __init__(self, name='timer', verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        # self.times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        dt = time.time() - self.start_time
        # self.times.append(dt)
        self.start_time = None
        if self.verbose:
            self.debug_print(dt)

    def debug_print(self, dt):
        print(f'{self.name:<10}: {int(dt*1000):>5} ms')
