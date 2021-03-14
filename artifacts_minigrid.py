# %%

import numpy as np
import gym
import gym_minigrid
import gym_minigrid.wrappers

CAT_TO_OBJ = gym_minigrid.wrappers.CategoricalObsWrapper(
    gym.make('MiniGrid-Empty-8x8-v0'),
    restrict_types=['basic', 'agent']  # TODO: this can vary, depending on setup
).possible_objects


def render_obs(obs,  # array(7,7),  categorical observation
               tile_size=16
               ):
    obs = CAT_TO_OBJ[obs]  # (7, 7, 3)

    # Find and remove special "agent" object
    agent_pos, agent_dir = None, None
    x, y = (obs[:, :, 0] == 10).nonzero()
    if len(x) > 0:
        agent_pos = x[0], y[0]  # In prediction there might be multiple agents, just pick first
        agent_dir = obs[x[0]][y[0]][2]
        obs[x, y, :] = np.array([1, 0, 0])  # Set agent pos to empty

    # Render
    grid, vis_mask = gym_minigrid.minigrid.Grid.decode(obs)
    img = grid.render(tile_size, agent_pos=agent_pos, agent_dir=agent_dir, highlight_mask=~vis_mask)
    return img  # (112, 112, RGB)
