import numpy as np
import scipy.signal

DISCOUNT_GAMMA = 0.99


def flatten(x):
    if x is None:
        return x
    return x.reshape([-1] + list(x.shape[2:]))


def discount(x: np.ndarray, gamma: float = DISCOUNT_GAMMA):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def return_cumulative(reward: np.ndarray, reset: np.ndarray):
    # PERF: would be better with numpy.ufunc.accumulate
    accumulate = []
    val = 0.0
    for i in range(len(reward)):
        val = val * (1 - reset[i]) + reward[i]
        accumulate.append(val)
    return np.array(accumulate)


def return_discounted(reward: np.ndarray, reset: np.ndarray, gamma=DISCOUNT_GAMMA):
    # PERF: would be better with numpy.ufunc.accumulate
    accumulate = []
    val0 = reward.mean() / (1.0 - DISCOUNT_GAMMA)
    val = val0
    accumulate.append(val)
    for i in reversed(range(len(reward) - 1)):
        val = gamma * val * (1 - reset[i + 1]) + reward[i + 1] + reset[i + 1] * val0
        accumulate.append(val)
    return np.array(accumulate)[::-1]


def _action_categorical(action):
    if action.dtype in [np.float32, np.float64]:
        return action.argmax(axis=-1)
    else:
        return action


def parse_batch_data(data, take_episodes=10):

    # Cut all to same B
    B = min(v.shape[0] for k, v in data.items())
    B = min(B, take_episodes)
    for k in data.keys():
        data[k] = data[k][0:B]  # Cut smaller
        if data[k].dtype == np.float16:
            data[k] = data[k].astype(np.float32)  # Operations are slow with float16

    # Pad all to same T
    T = max(v.shape[1] for k, v in data.items())
    for k in data.keys():
        v = data[k]
        t = v.shape[1]
        if t < T:
            data[k] = np.pad(v, ((0, 0), (0, T - t)), 'constant', constant_values=np.nan)

    i_batch, i_step = np.indices((B, T))
    i_batch_step = i_batch * 1000 + i_step  # type: ignore

    if data['image'].shape[-1] == 1:
        # Backwards-compatibility (7,7,1) => (7,7)
        data['image'] = data['image'][..., 0]
    
    if len(data['reward'].shape) == 3:
        # Categorical reward
        data['reward'] = np.argmax(data['reward'], -1)

    nans = np.full((data['reset'].shape), np.nan)

    ret = dict(
        step=flatten(i_batch_step),
        episode=flatten(i_batch),
        episode_step=flatten(i_step),
        #
        action=flatten(data['action']).argmax(axis=-1),
        reward=flatten(data['reward']),
        vecnovel=flatten(data.get('vecnovel', nans)),
        reset=flatten(data.get('reset', nans)),
        terminal=flatten(data.get('terminal', nans)),
        image=flatten(data['image']),
        map_agent=flatten(data.get('map_agent')),
        # map_centered=flatten(data.get('map_centered')),
        map=flatten(data.get('map')),
        agent_pos=flatten(data.get('agent_pos')),
        agent_dir=flatten(data.get('agent_dir')),
        goal_direction=flatten(data.get('goal_direction')),
        goals_direction=flatten(data.get('goals_direction')),
        goals_pos=flatten(data.get('targets_pos')),
        #
        image_rec=flatten(data.get('image_rec_p', data.get('image_rec'))),
        map_rec=flatten(data.get('map_rec_p', data.get('map_rec'))),
        goal_direction_pred=flatten(data.get('goal_direction_pred')),
        goals_direction_pred=flatten(data.get('goals_direction_pred')),
        #
        image_pred=flatten(data.get('image_pred_p', data.get('image_pred'))),
        reward_pred=flatten(data.get('reward_pred', nans)),
        terminal_pred=flatten(data.get('terminal_pred', nans)),
        action_pred=flatten(data['action_pred']).argmax(axis=-1) if 'action_pred' in data else flatten(nans),
        action_prob=flatten(data.get('action_prob', nans)),
        #
        loss_kl=flatten(data.get('loss_kl', nans)),
        loss_image=flatten(data.get('loss_image', nans)),
        loss_map=flatten(data.get('loss_map', nans)),
        loss_goal_direction=flatten(data.get('loss_goal_direction', nans)),
        acc_map=flatten(data.get('acc_map', nans) * 100.),
        logprob_image=flatten(data.get('logprob_image', nans)),
        entropy_prior=flatten(data.get('entropy_prior', nans)),
        entropy_post=flatten(data.get('entropy_post', nans)),
        #
        value=flatten(data.get('value', data.get('policy_value', nans))),
        value_target=flatten(data.get('value_target', nans)),
        value_advantage=flatten(data.get('value_advantage', nans)),
        value_advantage_gae=flatten(data.get('value_advantage_gae', nans)),
        value_weight=flatten(data.get('value_weight', nans)),
        value_aux=flatten(data.get('policy_value_aux', nans)),
    )
    ret.update({'return': return_cumulative(ret['reward'], ret['reset'])})
    
    ret = {k: v for k, v in ret.items() if v is not None}  # Remove Nones
    return ret


def parse_episode_data(data):
    n = data['reward'].shape[0]
    i_step = np.arange(n)

    if 'image' not in data and 'image_t' in data:
        data['image'] = data['image_t'].transpose(3, 0, 1, 2)  # CHWN => NCHW
        del data['image_t']

    if data['image'].shape[-1] == 1:
        # Backwards-compatibility (7,7,1) => (7,7)
        data['image'] = data['image'][..., 0]

    nans = np.full((data['reset'].shape), np.nan)

    ret = dict(
        step=i_step,
        action=_action_categorical(data['action']),
        reward=data['reward'],
        vecnovel=data.get('vecnovel', nans),
        image=data['image'],
        terminal=data.get('terminal', nans),
        reset=data.get('reset', nans),
        map_agent=data.get('map_agent'),
        map=data.get('map', data.get('maze_layout')),
        map_rec=data.get('map_centered'),
        agent_pos=data.get('agent_pos', nans),
        agent_dir=data.get('agent_dir', nans),
        goals_pos=data.get('targets_pos'),
        value=flatten(data.get('policy_value', nans)),
        **{
            'return': return_cumulative(data['reward'], data['reset']),
            'return_discounted': return_discounted(data['reward'], data['reset']),
        }
    )

    ret = {k: v for k, v in ret.items() if v is not None}  # Remove Nones
    return ret
