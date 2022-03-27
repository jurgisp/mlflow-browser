# %%

import numpy as np
import skimage.transform as skt

try:
    import gym
    import gym_minigrid
    import gym_minigrid.wrappers
    import gym_minigrid.minigrid

    CAT_TO_OBJ = gym_minigrid.wrappers.CategoricalObsWrapper(
        gym.make('MiniGrid-Empty-8x8-v0'),
        restrict_types=['basic', 'agent', 'box']  # TODO: this can vary, depending on setup
    ).possible_objects

    COLORS = list(gym_minigrid.minigrid.COLORS.values())

except:
    print('No gym_minigrid')


def rotation(ang):
    ang = np.radians(ang)
    return np.array((
        (np.cos(ang), -np.sin(ang)),   # type: ignore
        (np.sin(ang), np.cos(ang))
    ))

def rotation_dir(dir):
    assert len(dir) == 2
    return np.array((
        (dir[0], -dir[1]),
        (dir[1], dir[0])
    ))


def render_obs(obs, trajectory=None, agent_pos=None, agent_dir=None, goals_pos=None, tile_size=16):
    import skimage.draw

    def draw_line(img, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        ly, lx, val = skimage.draw.line_aa(int(16 * y1), int(16 * x1), int(16 * y2), int(16 * x2))
        img[ly, lx] = np.clip(img[ly, lx] + val[:, None] * np.array([255, 0, 0]), 0, 255)

    def draw_triangle(img, p1, p2, p3):
        triangle = np.array([p1, p2, p3])
        ly, lx = skimage.draw.polygon(triangle[:, 1] * 16, triangle[:, 0] * 16, shape=img.shape)
        img[ly, lx] = np.array([255, 0, 0])

    def draw_circle(img, p, radius=3, color=[255, 0, 0]):
        ly, lx = skimage.draw.disk((p[1] * 16, p[0] * 16), radius, shape=img.shape)
        img[ly, lx] = np.array(color)

    if obs is None:
        obs = np.zeros((7, 7), dtype=int)

    # (N,N) - sampled frame
    if len(obs.shape) == 2 and obs.shape[0] == obs.shape[1] and np.issubdtype(obs.dtype, np.integer):
        if obs.shape == (7, 7):
            # mark agent position on observation
            obs[3, 6] = 7

        img = _render_obs(obs)

        if trajectory and len(trajectory) > 1:
            # draw trajectory
            for i in range(1, len(trajectory)):
                draw_line(img, trajectory[i - 1], trajectory[i])

        if agent_pos is not None:
            # draw agent for maze3d
            pos = np.array(agent_pos)
            dir = np.array(agent_dir)
            draw_triangle(img,
                          pos + 0.4 * dir,
                          pos + 0.3 * rotation(120) @ dir,
                          pos + 0.3 * rotation(240) @ dir)

        if goals_pos is not None:
            # draw goal predictions
            for i, goal_pos in enumerate(goals_pos):
                draw_circle(img, goal_pos, color=COLORS[i])

        return img

    # (N,N,C) - probabilities
    if len(obs.shape) == 3 and obs.shape[0] == obs.shape[1] and not np.issubdtype(obs.dtype, np.integer):
        img = None
        for i in range(obs.shape[-1]):
            # Combine image filled by each category according to prob weights
            # TODO: this doesn't work for drawing agent, because _render_obs() can not fill the image with agents in each cell
            img_cat = _render_obs(np.full(obs.shape[: 2], i), tile_size=tile_size)
            weight_cat = np.repeat(np.repeat(obs[..., i].T, tile_size, axis=0), tile_size, axis=1)
            if img is None:
                img = np.zeros(img_cat.shape)
            img += np.expand_dims(weight_cat, -1) * img_cat
        assert img is not None
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
    map = np.rot90(map, (agent_dir + 1))
    mid = (map.shape[0] - 1) // 2
    top_x, top_y = mid - agent_pos[0], mid - agent_pos[1]
    map = map[top_x: top_x + size[0], top_y: top_y + size[0]]
    return map


def map_centric_to_global_rgb(map, agent_pos, agent_dir, size):
    angle = np.arctan2(agent_dir[0], agent_dir[1])
    map = skt.rotate(map, angle / np.pi * 180 + 180)

    res = map.shape[0] / size[0] / 2
    min_x = int(map.shape[0] / 2 - res * agent_pos[0])
    max_x = int(map.shape[0] - res * agent_pos[0])
    min_y = int(map.shape[1] / 2 - res * agent_pos[1])
    max_y = int(map.shape[1] - res * agent_pos[1])
    map = map[min_y: max_y, min_x: max_x]

    return map
