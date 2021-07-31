import numpy as np


def flatten(x):
    return x.reshape([-1] + list(x.shape[2:]))


# def parse_d2_train_batch(data):
#     b, t = data['reward'].shape
#     n = b * t
#     i_batch, i_step = np.indices((b, t))
#     i_batch_step = i_batch * 1000 + i_step

#     if data['imag_reward'].shape[1] == t - 1:
#         # Last step doesn't have imag_* when running with discount head
#         # Append zeros to make of the same length
#         for k in data.keys():
#             if k.startswith('imag_'):
#                 x = data[k]
#                 data[k] = np.concatenate([x, np.zeros_like(x[:, :1, ...])], axis=1)  # (B, T-1, ...) => (B, T, ...)

#     return dict(
#         step=flatten(i_batch_step),
#         action=flatten(data['action']).argmax(axis=-1),
#         reward=flatten(data['reward']),
#         image=flatten(data['image'])[..., 0],  # (7,7,1) => (7,7)
#         #
#         reward_rec=flatten(data['reward_rec']) if 'reward_rec' in data else [np.nan] * n,
#         image_rec=flatten(data['image_rec']),  # (7,7)
#         #
#         action_pred=flatten(data['imag_action']).argmax(axis=-1)[:, 0],  # imag_action[0] = <act1>
#         reward_pred=flatten(data['imag_reward'])[:, 1],                  # imag_reward[1] = reward(state(act1))
#         discount_pred=flatten(data['imag_weights'])[:, 1],               # imag_weights[1] = discount(state(act1))
#         image_pred=flatten(data['imag_image'])[:, 1],                    # imag_image[1] = image(state(act1))
#         #
#         value=flatten(data['imag_value'])[:, 0],                         # imag_value[0] = value(start)
#         value_target=flatten(data['imag_target'])[:, 0],                 # imag_target[0] = value_target(start)
#         loss_kl=flatten(data['loss_kl']) if 'loss_kl' in data else [np.nan] * n,
#     )


def parse_d2_wm_predict(data, take_episodes=1):
    for k in data.keys():
        data[k] = data[k][:take_episodes]
        if data[k].dtype == np.float16:
            data[k] = data[k].astype(np.float32)  # Operations are slow with float16

    b, t = data['reward'].shape
    n = b * t
    i_batch, i_step = np.indices((b, t))
    i_batch_step = i_batch * 1000 + i_step

    if data['image'].shape[-1] == 1:
        # Backwards-compatibility (7,7,1) => (7,7)
        data['image'] = data['image'][..., 0]

    nans = np.full_like(data['reward'], np.nan)
    noimg = np.zeros_like(data['image'])

    return dict(
        step=flatten(i_batch_step),
        episode=flatten(i_batch),
        episode_step=flatten(i_step),
        #
        action=flatten(data['action']).argmax(axis=-1),
        reward=flatten(data['reward']),
        terminal=flatten(data.get('terminal', nans)),
        image=flatten(data['image']),
        map_agent=flatten(data.get('map_agent', noimg)),
        map=flatten(data.get('map', noimg)),
        #
        image_rec=flatten(data.get('image_rec_p', data.get('image_rec', noimg))),
        map_rec=flatten(data.get('map_rec_p', data.get('map_rec', noimg))),
        #
        image_pred=flatten(data.get('image_pred_p', data.get('image_pred', noimg))),
        reward_pred=flatten(data.get('reward_pred', nans)),
        discount_pred=flatten(data.get('discount_pred', nans)),
        #
        loss_kl=flatten(data.get('loss_kl', nans)),
        loss_image=flatten(data.get('loss_image', nans)),
        loss_map=flatten(data.get('loss_map', nans)),
        acc_map=flatten(data.get('acc_map', nans)),
        logprob_img=flatten(data.get('logprob_img', nans)),
        entropy_prior=flatten(data.get('entropy_prior', nans)),
        entropy_post=flatten(data.get('entropy_post', nans)),
        #
        value=flatten(data.get('behav_value', nans)),
        action_pred=flatten(data['behav_action']).argmax(axis=-1) if 'behav_action' in data else np.array([np.nan] * n),
    )


def parse_d2_episodes(data):
    n = data['reward'].shape[0]
    i_step = np.arange(n)

    if 'image' not in data and 'image_t' in data:
        data['image'] = data['image_t'].transpose(3, 0, 1, 2)  # CHWN => NCHW
        del data['image_t']

    if data['image'].shape[-1] == 1:
        # Backwards-compatibility (7,7,1) => (7,7)
        data['image'] = data['image'][..., 0]

    nans = np.full_like(data['reward'], np.nan)
    noimg = np.zeros_like(data['image'])

    return dict(
        step=i_step,
        action=data['action'].argmax(axis=-1),
        reward=data['reward'],
        image=data['image'],
        terminal=data.get('terminal', nans),
        map_agent=data.get('map_agent', noimg),
        map=data.get('map', noimg),
        map_rec=data.get('map_centered', noimg),
        agent_pos=data.get('agent_pos', nans),
        agent_dir=data.get('agent_dir', nans),
    )
