import numpy as np
from scipy.signal import butter, lfilter


class ButterworthFilter:
    def __init__(
            self,
            order: int = 2,
            cutoff: float = 5.0,
            fs: float = 50.0,
            num_components: int = 6
        ):
        self.filters = [_ButterworthFilter(order, cutoff, fs) for _ in range(num_components)]

    def filter(self, new_values):
        return [f.filter(x) for f, x in zip(self.filters, new_values)]


class _ButterworthFilter:
    def __init__(self, order, cutoff, fs):
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        nyq = 0.5 * self.fs
        self.normal_cutoff = self.cutoff / nyq
        self.b, self.a = butter(self.order, self.normal_cutoff, btype='low', analog=False)
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def filter(self, x):
        y, self.zi = lfilter(self.b, self.a, [x], zi=self.zi)
        return y.tolist()[0]
