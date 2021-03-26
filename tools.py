import time
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Ignore Google Cloud SDK warning when running locally with user credentials
# This is used for downloading artifacts
warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")


def data_length(data):
    if data == {}:
        return 0
    any_key = next(iter(data.keys()))
    return len(data[any_key])

def selected_rows(src):
    ixs = src.selected.indices or []
    n = data_length(src.data)
    rows = [
        {key: src.data[key][i] for key in src.data}
        for i in ixs
        if i < n
    ]
    return rows


def selected_row_single(src):
    rows = selected_rows(src)
    return rows[0] if len(rows) == 1 else None


def selected_columns(src):
    ixs = src.selected.indices or []
    cols = {
        key: list([src.data[key][i] for i in ixs])
        for key in src.data
    }
    return cols


def download_artifact_npz(client, run_id, artifact_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, artifact_path, tmpdir)
        with Path(path).open('rb') as f:
            data = np.load(f)
            data = {k: data[k] for k in data.keys()}
    return data


def to_rgba(img, alpha=255):
    rgba = np.zeros(img.shape[0:2], dtype=np.uint32)
    view = rgba.view(dtype=np.uint8).reshape(rgba.shape + (4,))
    view[:, :, 0:3] = np.flipud(img)
    if isinstance(alpha, np.ndarray):
        view[:, :, 3] = np.flipud(alpha)
    else:
        view[:, :, 3] = alpha
    return rgba


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
