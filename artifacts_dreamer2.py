import numpy as np


def parse_d2_train_batch(data):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))

    b, t = data['reward'].shape
    i_batch, i_step = np.indices((b, t))
    i_batch_step = i_batch * 1000 + i_step

    if data['imag_reward'].shape[1] == t - 1:
        # Last step doesn't have imag_* when running with discount head
        # Append zeros to make of the same length
        for k in data.keys():
            if k.startswith('imag_'):
                x = data[k]
                data[k] = np.concatenate([x, np.zeros_like(x[:, :1, ...])], axis=1)  # (B, T-1, ...) => (B, T, ...)

    return dict(
        step=flatten(i_batch_step),
        action=flatten(data['action']).argmax(axis=-1),
        reward=flatten(data['reward']),
        image=flatten(data['image'])[..., 0],  # (7,7,1) => (7,7)
        image_rec=flatten(data['image_rec']),  # (7,7)
        reward_rec=flatten(data['reward_rec']) if 'reward_rec' in data else [np.nan] * (b * t),
        imag_action_1=flatten(data['imag_action']).argmax(axis=-1)[:, 0],
        # imag_reward_1=flatten(data['imag_reward'])[:, 0],
        # imag_image_1=flatten(data['imag_image'])[:, 0],
        imag_reward_2=flatten(data['imag_reward'])[:, 1],
        imag_image_2=flatten(data['imag_image'])[:, 1],
        imag_weights_2=flatten(data['imag_weights'])[:, 1],
        imag_value_1=flatten(data['imag_value'])[:, 0],
        imag_target_1=flatten(data['imag_target'])[:, 0],
        loss_kl=flatten(data['loss_kl']) if 'loss_kl' in data else [np.nan] * (b * t),
    )
