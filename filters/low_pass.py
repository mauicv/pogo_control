class LowPassFilter:
    def __init__(self, alpha=0.85, num_components=6):
        self.alpha = alpha
        self.prev_values = [None] * num_components

    def __call__(self, new_values):
        assert len(new_values) == len(self.prev_values), "Number of new values must match number of filters"
        for i in range(len(new_values)):
            if self.prev_values[i] is None:
                self.prev_values[i] = new_values[i]
            else:
                self.prev_values[i] = self.alpha * self.prev_values[i] + (1 - self.alpha) * new_values[i]
        return self.prev_values.copy()
