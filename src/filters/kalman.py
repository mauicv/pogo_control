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
