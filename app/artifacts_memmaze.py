import numpy as np

import skimage.draw

TILE = 16

TARGET_COLORS = [
    np.array([170, 38, 30]),  # red
    np.array([99, 170, 88]),  # green
    np.array([39, 140, 217]),  # blue
    np.array([93, 105, 199]),  # purple
    np.array([220, 193, 59]),  # yellow
    np.array([220, 128, 107]),  # salmon
]

def preprocess_frames_memmaze(sd):
    data = sd.copy()
    # data['map'] = np.flipud(sd['map'][..., None] * 127)

    data['map'] = render_map_probe_target(sd['map'])

    if 'map_rec' in sd:
        # data['map_rec'] = render_map_probe_target(sd['map_rec'].argmax(-1))
        data['map_rec'] = render_map_prediction_correct(sd['map_rec'].argmax(-1), sd['map'])

    data['map_agent'] = render_map_with_agent(
        sd['map'],
        trajectory=sd.get('agent_trajectory'),
        agent_pos=sd.get('agent_pos'),
        agent_dir=sd.get('agent_dir'),
        goals_pos=sd.get('goals_pos')
    )

    return data


def render_map_probe_target(map):
    # (N,N) - categorical image
    assert len(map.shape) == 2 and map.shape[0] == map.shape[1] and np.issubdtype(map.dtype, np.integer)
    img = _render_map(map, gridlines=True)
    img = np.flipud(img)
    return img


def render_map_prediction_correct(map, map_correct):
    # (N,N) - categorical image
    assert len(map.shape) == 2 and map.shape[0] == map.shape[1] and np.issubdtype(map.dtype, np.integer)
    assert map_correct is not None

    is_correct = (map == map_correct)
    is_wall = (map_correct == 0)
    mask_correct = np.repeat(np.repeat(is_correct, TILE, axis=0), TILE, axis=1)[..., None]
    mask_wall = np.repeat(np.repeat(is_wall, TILE, axis=0), TILE, axis=1)[..., None]
    img = (
        mask_correct * ~mask_wall * (100, 200, 100) +
        ~mask_correct * ~mask_wall * (200, 100, 100) +
        mask_correct * mask_wall * (50, 100, 50) +
        ~mask_correct * mask_wall * (100, 50, 50)
    ).astype(np.uint8)

    _make_grid_lines(img)

    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.flipud(img)
    return img


def render_map_with_agent(obs,
                          trajectory=None,
                          agent_pos=None,
                          agent_dir=None,
                          goals_pos=None):

    # (N,N) - categorical image
    assert len(obs.shape) == 2 and obs.shape[0] == obs.shape[1] and np.issubdtype(obs.dtype, np.integer)

    img = _render_map(obs, color_floor=(100, 100, 100), color_wall=(81, 57, 44))

    if trajectory and len(trajectory) > 1:
        # draw trajectory
        for i in range(1, len(trajectory)):
            draw_line(img,
                      np.array(trajectory[i - 1]),
                      np.array(trajectory[i]),
                      color=(255, 255, 255),
                      thickness=3)

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
            draw_circle(img, goal_pos, radius=5, color=TARGET_COLORS[i])

    img = np.flipud(img)
    return img


def _render_map(obs, color_floor=(255, 255, 255), color_wall=(0, 0, 0), gridlines=False):
    """Render probe in black and white."""
    is_floor = (obs == 1).astype(int)  # (9,9)
    is_wall = (obs == 0).astype(int)
    mask_floor = np.repeat(np.repeat(is_floor, TILE, axis=0), TILE, axis=1)[..., None]  # (144,144,1)
    mask_wall = np.repeat(np.repeat(is_wall, TILE, axis=0), TILE, axis=1)[..., None]
    img = (mask_floor * color_floor + mask_wall * color_wall).astype(np.uint8)  # (144,144,3)
    if gridlines:
        _make_grid_lines(img)
    return img


def _make_grid_lines(img):
    for i in np.arange(0, img.shape[0], TILE):
        img[i, :, :] = 0
    for i in np.arange(0, img.shape[1], TILE):
        img[:, i, :] = 0
    img[-1, :, :] = 0
    img[:, -1, :] = 0


def draw_line(img, p1, p2, thickness=3, color=(255, 255, 255)):
    for dx in (np.arange(thickness) - thickness // 2):
        for dy in (np.arange(thickness) - thickness // 2):
            x1 = int(TILE * p1[0]) + dx
            y1 = int(TILE * p1[1]) + dy
            x2 = int(TILE * p2[0]) + dx
            y2 = int(TILE * p2[1]) + dy
            ly, lx, mask = skimage.draw.line_aa(y1, x1, y2, x2)
            img[ly, lx] = np.clip(
                (1 - mask[:, None]) * img[ly, lx] + mask[:, None] * color,
                0, 255)


def draw_triangle(img, p1, p2, p3,):
    points = np.array([p1, p2, p3]) * TILE
    ly, lx = skimage.draw.polygon(points[:, 1], points[:, 0], shape=img.shape)
    img[ly, lx] = np.array([255, 255, 255])


def draw_circle(img, p, radius=3, color=[255, 0, 0]):
    ly, lx = skimage.draw.disk((p[1] * TILE, p[0] * TILE), radius, shape=img.shape)
    img[ly, lx] = np.array(color)


def rotation(ang):
    ang = np.radians(ang)
    return np.array((
        (np.cos(ang), -np.sin(ang)),   # type: ignore
        (np.sin(ang), np.cos(ang))
    ))
