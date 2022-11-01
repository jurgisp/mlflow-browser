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

    if 'map' not in sd:
        return data

    data['map'] = render_map_probe_target(sd['map'])

    if 'map_rec' in sd:
        # data['map_rec'] = render_map_probe_target(sd['map_rec'].argmax(-1))
        data['map_rec'] = render_map_predcorrect_redgreen(sd['map_rec'].argmax(-1), sd['map'])
        # data['map_rec'] = render_map_predcorrect_whitered(sd['map_rec'].argmax(-1), sd['map'])

    if 'goals_direction_pred' in sd:
        goals_direction = sd['goals_direction_pred'].reshape((-1, 2))
        goals_pred_pos = []
        agent_angle = np.degrees(np.arctan2(sd['agent_dir'][1], sd['agent_dir'][0]))
        for gd in goals_direction:
            goals_pred_pos.append(sd['agent_pos'] + rotation(agent_angle - 90.0) @ gd)
    else:
        goals_pred_pos = None

    data['map_agent'] = render_map_with_agent(
        sd['map'],
        trajectory=sd.get('agent_trajectory'),
        agent_pos=sd.get('agent_pos'),
        agent_dir=sd.get('agent_dir'),
        goals_pos=sd.get('goals_pos'),
        goals_pred_pos=goals_pred_pos,
    )

    return data


def render_map_probe_target(map):
    # (N,N) - categorical image
    assert len(map.shape) == 2 and map.shape[0] == map.shape[1] and np.issubdtype(map.dtype, np.integer)
    img = _render_map(map, gridlines=(255,))
    img = np.flipud(img)
    return img


def render_map_predcorrect_whitered(map, map_correct):
    # (N,N) - categorical image
    assert len(map.shape) == 2 and map.shape[0] == map.shape[1] and np.issubdtype(map.dtype, np.integer)
    assert map_correct is not None

    img = _render_map(map, gridlines=(255,))

    MARK_RED = (200, 100, 100)
    THICKNESS = 2

    is_wrong = (map != map_correct)
    iys, ixs = np.where(is_wrong)
    for iy, ix in zip(iys, ixs):
        draw_square_border(img,
                           ix * TILE + 1,
                           iy * TILE + 1,
                           ix * TILE + TILE - 1,
                           iy * TILE + TILE - 1,
                           thickness=THICKNESS,
                           color=MARK_RED)

    img = np.flipud(img)
    return img


def render_map_predcorrect_redgreen(map, map_correct):
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
                          goals_pos=None,
                          goals_pred_pos=None,
                          color_floor=(100, 100, 100), 
                          color_wall=(81, 57, 44),
                          ):

    # (N,N) - categorical image
    assert len(obs.shape) == 2 and obs.shape[0] == obs.shape[1] and np.issubdtype(obs.dtype, np.integer)

    img = _render_map(obs, color_floor=color_floor, color_wall=color_wall)

    if trajectory and len(trajectory) > 1:
        # draw trajectory
        for i in range(1, len(trajectory)):
            draw_line(img,
                      np.array(trajectory[i - 1]) * TILE,
                      np.array(trajectory[i]) * TILE,
                      color=(255, 255, 255),
                      thickness=2)

    if agent_pos is not None:  # 926454
        # draw agent for maze3d
        pos = np.array(agent_pos)
        dir = np.array(agent_dir)
        sz = 2.0
        draw_triangle(img,
                      (pos + 0.4 * sz * dir) * TILE,
                      (pos + 0.3 * sz * rotation(120) @ dir) * TILE,
                      (pos + 0.3 * sz * rotation(240) @ dir) * TILE)

    if goals_pos is not None:
        # draw goal true locations
        for i, goal_pos in enumerate(goals_pos):
            draw_disk(img, * tuple(goal_pos * TILE), radius=6, color=TARGET_COLORS[i])

    if goals_pred_pos is not None:
        # draw goal predictions
        for i, goal_pos in enumerate(goals_pred_pos):
            draw_cross(img, * tuple(goal_pos * TILE), size=6, thickness=2, color=TARGET_COLORS[i]*255//220)

    img = np.flipud(img)
    return img


def _render_map(obs, color_floor=(255, 255, 255), color_wall=(0, 0, 0), gridlines=None):
    """Render probe in black and white."""
    is_floor = (obs == 1).astype(int)  # (9,9)
    is_wall = (obs == 0).astype(int)
    mask_floor = np.repeat(np.repeat(is_floor, TILE, axis=0), TILE, axis=1)[..., None]  # (144,144,1)
    mask_wall = np.repeat(np.repeat(is_wall, TILE, axis=0), TILE, axis=1)[..., None]
    img = (mask_floor * color_floor + mask_wall * color_wall).astype(np.uint8)  # (144,144,3)
    if gridlines:
        _make_grid_lines(img, color=gridlines)
    return img


def _make_grid_lines(img, color=(0, 0, 0)):
    color = np.array(color)
    for i in np.arange(0, img.shape[0], TILE):
        img[i, :, :] = color
    for i in np.arange(0, img.shape[1], TILE):
        img[:, i, :] = color
    img[-1, :, :] = color
    img[:, -1, :] = color


def draw_line(img, p1, p2, thickness=1, color=(255, 255, 255)):
    for dx in (np.arange(thickness) - thickness // 2):
        for dy in (np.arange(thickness) - thickness // 2):
            x1 = int(p1[0]) + dx
            y1 = int(p1[1]) + dy
            x2 = int(p2[0]) + dx
            y2 = int(p2[1]) + dy
            ly, lx, mask = skimage.draw.line_aa(y1, x1, y2, x2)
            ly, lx, mask = clip_bounds(ly, lx, mask, img.shape)
            img[ly, lx] = np.clip(
                (1 - mask[:, None]) * img[ly, lx] + mask[:, None] * color,
                0, 255)

def clip_bounds(ly, lx, mask, shape):
    inbounds = (0 <= ly) & (ly < shape[0]) & (0 <= lx) & (lx < shape[1])
    return ly[inbounds], lx[inbounds], mask[inbounds]

def draw_triangle(img, p1, p2, p3):
    points = np.array([p1, p2, p3])
    ly, lx = skimage.draw.polygon(points[:, 1], points[:, 0], shape=img.shape)
    img[ly, lx] = np.array([255, 255, 255])


def draw_square_border(img, x1, y1, x2, y2, thickness=1, color=[255, 0, 0]):
    for d in range(thickness):
        ly, lx = skimage.draw.polygon_perimeter(
            [y1 + d, y1 + d, y2 - d, y2 - d],
            [x1 + d, x2 - d, x2 - d, x1 + d],
            shape=img.shape)
        img[ly, lx] = np.array(color)


def draw_disk(img, x, y, radius=3, color=[255, 0, 0]):
    ly, lx = skimage.draw.disk((y, x), radius, shape=img.shape)
    img[ly, lx] = np.array(color)


# def draw_circle(img, p, radius=3, color=[255, 0, 0]):
#     ly, lx = skimage.draw.circle_perimeter(int(p[1] * TILE), int(p[0] * TILE), radius, shape=img.shape)
#     img[ly, lx] = np.array(color)

def draw_cross(img, x, y, size=3, color=[255, 0, 0], thickness=1):
    draw_line(img, (x - size, y - size), (x + size, y + size), color=color, thickness=thickness)
    draw_line(img, (x - size, y + size), (x + size, y - size), color=color, thickness=thickness)


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
