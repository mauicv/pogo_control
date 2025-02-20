from filterpy.kalman import KalmanFilter
import numpy as np


def make_ds_filter(d, speed):
    filter = KalmanFilter(dim_x=2, dim_z=1)
    filter.x = np.array([d, speed])
    filter.F = np.array([[1, 1], [0, 1]])
    filter.H = np.array([[1, 0]])
    filter.P = np.eye(2) * 1000
    filter.R = np.eye(1) * 5
    filter.Q = np.array([
        [1, 0],
        [0, 25]
    ])
    return filter


class KalmanDSFilter:
    def __init__(self, d):
        self.filter = make_ds_filter(d, 0)

    def __call__(self, d):
        self.filter.predict()
        self.filter.update(np.array([d]))
        return self.filter.x[0], self.filter.x[1]
    
    @property
    def x(self):
        return self.filter.x


def make_xv_kalman_filter(init_x, init_y, init_vx, init_vy):
    filter = KalmanFilter(dim_x=4, dim_z=2)
    filter.x = np.array([init_x, init_y, init_vx, init_vy])
    filter.F = np.array([
        [1, 0, 0.1, 0],
        [0, 1, 0, 0.1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    filter.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    filter.P = np.eye(4) * 1000
    filter.R = np.eye(2) * 5
    filter.Q = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 25, 0],
        [0, 0, 0, 25]
    ])
    return filter


class KalmanXVFilter:
    def __init__(self, init_x, init_y):
        self.filter = make_xv_kalman_filter(init_x, init_y, 0, 0)

    def __call__(self, x, y):
        self.filter.predict()
        self.filter.update(np.array([x, y]))
        x, y, vx, vy = self.filter.x
        return (x, y), (vx, vy)

    @property
    def x(self):
        return self.filter.x
