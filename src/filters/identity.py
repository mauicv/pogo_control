class IdentityFilter:
    def __init__(
            self,
            **kwargs
        ):
        pass

    def __call__(self, new_values):
        return new_values.tolist()
    
    def reset(self):
        pass
