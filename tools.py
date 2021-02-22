import pandas as pd


def metrics_to_df(metric_history):
    return pd.DataFrame(
        [
            [m.key, m.timestamp, m.step, m.value]
            for m in metric_history
        ],
        columns=['metric', 'timestamp', 'step', 'value']
    )
