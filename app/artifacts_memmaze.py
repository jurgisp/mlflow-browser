import numpy as np

def preprocess_frames_memmaze(sd):
    data = sd.copy()
    data['map'] = np.flipud(sd['map'][..., None] * 127)
    data['map_agent'] = data['map'].copy()
    return data
