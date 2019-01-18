import numpy as np
import torch

def to_torch(x, device='cuda'):
    if x is None:
        return None
    elif type(x) == dict:
        return { k: to_torch(v) for k, v in x.items() }
    elif type(x) in [list, tuple]:
        return [to_torch(v) for v in x]
    return torch.from_numpy(x).to(device)

def from_torch(t):
    if type(t) == dict:
        return { k: from_torch(v) for k, v in t.items() if v is not None }
    elif type(t) in [list, tuple]:
        return [from_torch(v) for v in t]
    
    x = t.detach().cpu().numpy()
    if x.size == 1 or np.isscalar(x):
        return np.asscalar(x)
    return x