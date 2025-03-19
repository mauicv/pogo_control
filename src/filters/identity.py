import numpy as np
import torch

class IdentityFilter:
    def __init__(
            self,
            **kwargs
        ):
        pass

    def __call__(self, new_values):
        if isinstance(new_values, (np.ndarray, torch.Tensor)):
            return new_values.tolist()
        return new_values
    
    def reset(self):
        pass
