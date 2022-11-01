import tempfile
from pathlib import Path

import numpy as np

from .artifacts_npz import parse_batch_data, parse_episode_data
from .tools import Timer


def render_step_frames(step_data=None,
                       image_keys=['image', 'image_rec', 'image_pred', 'map', 'map_agent', 'map_rec']
                       ):
    if step_data is None:
        return {k: [] for k in image_keys}

    sd = step_data.copy()

    TYPE = ''  # TODO: detect automatically

    if TYPE == 'maze2d':
        from .artifacts_minigrid import preprocess_frames_maze2d
        sd = preprocess_frames_maze2d(step_data)
    
    if TYPE == 'maze3d':
        from .artifacts_minigrid import preprocess_frames_maze3d
        sd = preprocess_frames_maze3d(step_data)
    
    if TYPE == 'memmaze':
        from .artifacts_memmaze import preprocess_frames_memmaze
        sd = preprocess_frames_memmaze(step_data)

    # Convert numpy arrays to Bokeh-compatible images

    data = {}
    for k in image_keys:
        img = sd.get(k)
        if img is not None and len(img.shape) == 3 and img.shape[-1] in (1, 3):
            # Looks like an image
            data[k] = [to_rgba(img)]
        else:
            # Something else
            if img is not None:
                print(f'Could not render {k}: {img.shape}')
            data[k] = [to_rgba(np.zeros((1, 1, 3)))]  # blank

    return data

    # if 'map_rec' in sd and 'map_agent' in sd and not np.all(sd['map_rec'] == 0):
    #     if len(sd['map_rec'].shape) == 2 and sd['map_rec'].shape[0] > sd['map_agent'].shape[0]:
    #         # map_rec is bigger than map_agent (categorical) - must be agent-centric
    #         map_agent = artifacts_minigrid.CAT_TO_OBJ[sd['map_agent']]
    #         agent_pos, agent_dir = artifacts_minigrid._get_agent_pos(map_agent)
    #         sd['map_rec_global'] = artifacts_minigrid._map_centric_to_global(sd['map_rec'], agent_pos, agent_dir, map_agent.shape[:2])
    #     if len(sd['map_rec'].shape) == 3 and sd['map_rec'].shape[-1] == 3 and 'agent_pos' in sd:
    #         # map_rec is RGB - must be agent centric
    #         sd['map_rec_global'] = artifacts_minigrid.map_centric_to_global_rgb(sd['map_rec'], sd['agent_pos'], sd['agent_dir'], sd['map_agent'].shape[:2])



def to_rgba(img, alpha=255):
    if img.min() < 0:  # (-0.5,0.5)
        img = img + 0.5
    if img.max() < 1.01:  # (0,1)
        img = img * 255
    img = img.clip(0, 255).astype(np.uint8)

    rgba = np.zeros(img.shape[0:2], dtype=np.uint32)
    view = rgba.view(dtype=np.uint8).reshape(rgba.shape + (4,))
    view[:, :, 0:3] = np.flipud(img)
    if isinstance(alpha, np.ndarray):
        view[:, :, 3] = np.flipud(alpha)
    else:
        view[:, :, 3] = alpha
    return rgba


def load_artifact_steps(mlclient, run_id, artifact_path, fill_trajectory=False):
    with Timer(f'mlflow.download_artifact({artifact_path})', verbose=True):
        if artifact_path.endswith('.npz'):
            data = download_artifact_npz(mlclient, run_id, artifact_path)
        else:
            print(f'Artifact extension not supported: {artifact_path}')
            return {}

    print('Artifact raw: ' + str({k: v.shape for k, v in data.items()}))  # type: ignore

    dimensions = len(data['reward'].shape)

    if dimensions == 1:
        # Looks like episode
        data_parsed = parse_episode_data(data)

    elif dimensions == 2:
        # Looks like batch
        data_parsed = parse_batch_data(data)

    else:
        data_parsed = {}
        print(f'Artifact type not supported: {artifact_path}')

    print('Artifact parsed: ' + str({k: v.shape for k, v in data_parsed.items()}))

    # Create agent_trajectory
    if fill_trajectory:
        agent_pos = data_parsed['agent_pos']  # maze3d
        steps = data_parsed['step']
        if np.any(np.isnan(agent_pos)):
            steps, ax, ay = np.where(data_parsed['map_agent'] >= 4)
            agent_pos = 0.5 + np.array([ax, ay]).T  # maze2d
        agent_trajectory = [list() for _ in steps]
        for i in steps:
            agent_trajectory[i] = agent_pos[:i + 1].tolist()
        data_parsed['agent_trajectory'] = agent_trajectory

    return data_parsed


def download_artifact_npz(client, run_id, artifact_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = client.download_artifacts(run_id, artifact_path, tmpdir)
        with Path(path).open('rb') as f:
            data = np.load(f)
            data = {k: data[k] for k in data.keys()}
    return data
