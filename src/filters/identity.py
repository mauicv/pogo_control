import numpy as np


class IdentityFilter:
    def __init__(
            self,
            **kwargs
        ):
        pass

    def __call__(self, new_values):
        if isinstance(new_values, np.ndarray):
            return new_values.tolist()
        return new_values
    
    def reset(self):
        pass
