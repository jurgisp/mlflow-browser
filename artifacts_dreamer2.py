import numpy as np

def flatten(x):
    return x.reshape([-1] + list(x.shape[2:]))


def parse_d2_train_batch(data):
    b, t = data['reward'].shape
    n = b * t
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
        #
        reward_rec=flatten(data['reward_rec']) if 'reward_rec' in data else [np.nan] * n,
        image_rec=flatten(data['image_rec']),  # (7,7)
        #
        action_pred=flatten(data['imag_action']).argmax(axis=-1)[:, 0],  # imag_action[0] = <act1>
        reward_pred=flatten(data['imag_reward'])[:, 1],                  # imag_reward[1] = reward(state(act1))
        discount_pred=flatten(data['imag_weights'])[:, 1],               # imag_weights[1] = discount(state(act1))
        image_pred=flatten(data['imag_image'])[:, 1],                    # imag_image[1] = image(state(act1))
        #
        value=flatten(data['imag_value'])[:, 0],                         # imag_value[0] = value(start)
        value_target=flatten(data['imag_target'])[:, 0],                 # imag_target[0] = value_target(start)
        loss_kl=flatten(data['loss_kl']) if 'loss_kl' in data else [np.nan] * n,
    )


def parse_d2_wm_predict(data):
    b, t = data['reward'].shape
    n = b * t
    i_batch, i_step = np.indices((b, t))
    i_batch_step = i_batch * 1000 + i_step

    return dict(
        step=flatten(i_batch_step),
        action=flatten(data['action']).argmax(axis=-1),
        reward=flatten(data['reward']),
        image=flatten(data['image'])[..., 0],  # (7,7,1) => (7,7)
        #
        image_pred=flatten(data['image_pred']),
        reward_pred=flatten(data['reward_pred']),
        discount_pred=flatten(data['discount_pred']),
        #
        value=flatten(data['behav_value']) if 'behav_value' in data else [np.nan] * n,
        action_pred=flatten(data['behav_action']).argmax(axis=-1) if 'behav_action' in data else [np.nan] * n,
    )
