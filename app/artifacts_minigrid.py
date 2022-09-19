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


def render_obs(obs,
               map_correct=None, 
               trajectory=None, 
               agent_pos=None, 
               agent_dir=None, 
               goals_pos=None,
               tile_size=16, 
               is_probe_target=False, 
               is_maze3d=False):
    import skimage.draw

    def draw_line(img, p1, p2, thickness=3, color=(255, 255, 255)):
        for dx in (np.arange(thickness) - thickness // 2):
            for dy in (np.arange(thickness) - thickness // 2):
                x1 = int(16 * p1[0]) + dx
                y1 = int(16 * p1[1]) + dy
                x2 = int(16 * p2[0]) + dx
                y2 = int(16 * p2[1]) + dy
                ly, lx, mask = skimage.draw.line_aa(y1, x1, y2, x2)
                img[ly, lx] = np.clip(
                    (1 - mask[:, None]) * img[ly, lx] + mask[:, None] * color,
                    0, 255)

    def draw_triangle(img, p1, p2, p3):
        points = np.array([p1, p2, p3]) * 16
        ly, lx = skimage.draw.polygon(points[:, 1], points[:, 0])
        img[ly, lx] = np.array([255, 255, 255])
        border_color = (0, 0, 0)
        # draw_line(img, p1, p2, color=border_color)
        # draw_line(img, p2, p3, color=border_color)
        # draw_line(img, p3, p1, color=border_color)

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

        if is_probe_target:
            img = _render_obs_colors(obs, gridlines=True)
        elif is_maze3d:
            img = _render_obs_colors(obs, color_floor=(100, 100, 100), color_wall=(81, 57, 44))
        else:
            img = _render_obs(obs)

        if trajectory and len(trajectory) > 1:
            # draw trajectory
            for i in range(1, len(trajectory)):
                off = 0.0 if is_maze3d else 0.5
                draw_line(img,
                          np.array(trajectory[i - 1]) + off,
                          np.array(trajectory[i]) + off,
                          color=(255, 255, 255) if is_maze3d else (255, 0, 0),
                          thickness=3 if is_maze3d else 2)

        if agent_pos is not None:  # 926454
            # draw agent for maze3d
            pos = np.array(agent_pos)
            dir = np.array(agent_dir)
            sz = 2.0
            draw_triangle(img,
                          pos + 0.4 * sz * dir,
                          pos + 0.3 * sz * rotation(120) @ dir,
                          pos + 0.3 * sz * rotation(240) @ dir)

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
            if i == 1:
                img_cat *= 0
            if i == 3:
                img_cat[:, :] = np.array([0, 0, 255])  # HACK: change goal to blue, so we can see it among the tint
            weight_cat = np.repeat(np.repeat(obs[..., i].T, tile_size, axis=0), tile_size, axis=1)
            if img is None:
                img = np.zeros(img_cat.shape)
            img += np.expand_dims(weight_cat, -1) * img_cat

        if map_correct is not None:
            p_correct = (obs * np.eye(obs.shape[-1])[map_correct]).sum(-1)
            is_correct = (p_correct > 0.50)
            is_wall = (map_correct == 2)
            mask_correct = np.repeat(np.repeat(is_correct.T, tile_size, axis=0), tile_size, axis=1)[..., None]
            mask_wall = np.repeat(np.repeat(is_wall.T, tile_size, axis=0), tile_size, axis=1)[..., None]
            img = (
                # mask_correct * mask_wall * (150, 255, 150) +
                # ~mask_correct * mask_wall * (255, 150, 150) +
                # mask_correct * ~mask_wall * (0, 64, 0) +
                # ~mask_correct * ~mask_wall * (100, 0, 0)
                mask_correct * ~mask_wall * (100, 200, 100) +
                ~mask_correct * ~mask_wall * (200, 100, 100) +
                mask_correct * mask_wall * (50, 100, 50) +
                ~mask_correct * mask_wall * (100, 50, 50)
            ).astype(np.uint8)
            _make_grid_lines(img, tile_size)

        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    assert False, f'Shape {obs.shape} ({obs.dtype}) not like MiniGrid'


def _render_obs(obs, tile_size=16):
    obs = CAT_TO_OBJ[obs]  # (N,N) => (N,N,3)
    agent_pos, agent_dir = _get_agent_pos(obs, remove_from_map=True)
    grid, vis_mask = gym_minigrid.minigrid.Grid.decode(obs)
    img = grid.render(tile_size, agent_pos=agent_pos, agent_dir=agent_dir, highlight_mask=~vis_mask)
    return img  # (112, 112, RGB)


def _render_obs_colors(obs, tile_size=16, color_floor=(255, 255, 255), color_wall=(0, 0, 0), gridlines=False):
    """Render probe in black and white."""
    obs = CAT_TO_OBJ[obs]  # (9,9) => (9,9,3)
    is_floor = (obs[..., 0] == 1).astype(int)  # (9,9)
    is_wall = (obs[..., 0] == 2).astype(int)
    mask_floor = np.repeat(np.repeat(is_floor.T, tile_size, axis=0), tile_size, axis=1)[..., None]  # (144,144,1)
    mask_wall = np.repeat(np.repeat(is_wall.T, tile_size, axis=0), tile_size, axis=1)[..., None]
    img = (mask_floor * color_floor + mask_wall * color_wall).astype(np.uint8)  # (144,144,3)
    if gridlines:
        _make_grid_lines(img, tile_size)
    return img


def _make_grid_lines(img, tile_size):
    for i in np.arange(0, img.shape[0], tile_size):
        img[i, :, :] = 0
    for i in np.arange(0, img.shape[1], tile_size):
        img[:, i, :] = 0


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
