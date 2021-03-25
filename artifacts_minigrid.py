# %%

import numpy as np
import gym
import gym_minigrid
import gym_minigrid.wrappers
from tools import Timer

CAT_TO_OBJ = gym_minigrid.wrappers.CategoricalObsWrapper(
    gym.make('MiniGrid-Empty-8x8-v0'),
    restrict_types=['basic', 'agent']  # TODO: this can vary, depending on setup
).possible_objects


def render_obs(obs, tile_size=16):
    if obs is None:
        obs = np.zeros((7, 7), dtype=int)

    # (N,N) - sampled frame
    if len(obs.shape) == 2 and obs.shape[0] == obs.shape[1] and np.issubdtype(obs.dtype, np.integer):
        return _render_obs(obs)

    # (N,N,C) - probabilities
    if len(obs.shape) == 3 and obs.shape[0] == obs.shape[1] and not np.issubdtype(obs.dtype, np.integer):
        img = None
        for i in range(obs.shape[-1]):
            # Combine image filled by each category according to prob weights
            img_cat = _render_obs(np.full(obs.shape[:2], i), tile_size=tile_size)
            weight_cat = np.repeat(np.repeat(obs[..., i].T, tile_size, axis=0), tile_size, axis=1)
            if img is None:
                img = np.zeros(img_cat.shape)
            img += np.expand_dims(weight_cat, -1) * img_cat
        return img.astype(np.uint8)

    assert False, f'Shape {obs.shape} ({obs.dtype}) not like MiniGrid'


def _render_obs(obs, tile_size=16):
    obs = CAT_TO_OBJ[obs]  # (N,N) => (N,N,3)
    agent_pos, agent_dir = _get_agent_pos(obs, remove_from_map=True)
    grid, vis_mask = gym_minigrid.minigrid.Grid.decode(obs)
    img = grid.render(tile_size, agent_pos=agent_pos, agent_dir=agent_dir, highlight_mask=~vis_mask)
    return img  # (112, 112, RGB)


def _get_agent_pos(map, remove_from_map=False):
    # Find and remove special "agent" object
    agent_pos, agent_dir = None, None
    x, y = (map[:, :, 0] == 10).nonzero()
    if len(x) > 0:
        agent_pos = x[0], y[0]  # In prediction there might be multiple agents, just pick first
        agent_dir = map[x[0]][y[0]][2]
        if remove_from_map:
            map[x, y, :] = np.array([1, 0, 0])  # Set agent pos to empty
    return agent_pos, agent_dir


def _map_centric_to_global(map, agent_pos, agent_dir, size):
    map = np.rot90(map, (agent_dir+1))
    mid = (map.shape[0] - 1) // 2
    top_x, top_y = mid - agent_pos[0], mid - agent_pos[1]
    map = map[top_x:top_x + size[0], top_y:top_y + size[0]]
    return map
