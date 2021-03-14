import numpy as np


def parse_d2_train_batch(data):
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))

    batch, step = np.indices(data['reward'].shape)
    return dict(
        batch=flatten(batch),
        step=flatten(step),
        action=flatten(data['action']).argmax(axis=-1),
        reward=flatten(data['reward']),
        image=flatten(data['image'])[..., 0],  # (7,7,1) => (7,7)
        image_rec=flatten(data['image_rec']),  # (7,7)
        imag_action_1=flatten(data['imag_action']).argmax(axis=-1)[:, 0],
        imag_reward_1=flatten(data['imag_reward'])[:, 0],
        imag_reward_2=flatten(data['imag_reward'])[:, 1],
        imag_value_1=flatten(data['imag_value'])[:, 0],
        imag_target_1=flatten(data['imag_target'])[:, 0],
        imag_image_1=flatten(data['imag_image'])[:, 0],
        imag_image_2=flatten(data['imag_image'])[:, 1],
    )
