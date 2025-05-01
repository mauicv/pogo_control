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


def make_xyzv_kalman_filter(init_x, init_y, init_z):
    filter = KalmanFilter(dim_x=3, dim_z=3)
    filter.x = np.array([init_x, init_y, init_z])
    filter.F = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    filter.H = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    filter.P = np.eye(3) * 1000
    filter.R = np.eye(3) * 5
    filter.Q = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    return filter


class KalmanXYZFilter:
    def __init__(self, init_x, init_y, init_z):
        self.filter = make_xyzv_kalman_filter(init_x, init_y, init_z)

    def __call__(self, x, y, z):
        self.filter.predict()
        self.filter.update(np.array([x, y, z]))
        x, y, z = self.filter.x
        return x, y, z

    @property
    def x(self):
        return self.filter.x


def make_1D_kalman_filter(val):
    filter = KalmanFilter(dim_x=1, dim_z=1)
    filter.x = np.array([val])
    filter.F = np.array([[1]])
    filter.H = np.array([[1]])
    filter.P = np.eye(1) * 1000
    filter.R = np.eye(1) * 5
    filter.Q = np.array([[1]])
    return filter


class KalmanYawFilter:
    def __init__(self, init_yaw):
        self.filter = make_1D_kalman_filter(init_yaw)
    
    def __call__(self, yaw):
        self.filter.predict()
        self.filter.update(np.array([yaw]))
        return self.filter.x[0]

    @property
    def x(self):
        return self.filter.x


def make_mpu6050_filter():
    filter = KalmanFilter(dim_x=6, dim_z=6)
    filter.x = np.array([0, 0, 0, 0, 0, 0])
    filter.F = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    filter.H = np.eye(6)
    filter.P = np.eye(6) * 1000
    filter.R = np.eye(6) * 5
    filter.Q = np.eye(6) * 0.5
    return filter


class KalmanMPU6050Filter:
    def __init__(self, ):
        self.filter = make_mpu6050_filter()

    def __call__(self, data):
        ax, ay, az, gx, gy, gz = data
        self.filter.predict()
        self.filter.update(np.array([ax, ay, az, gx, gy, gz]))
        return self.filter.x
    

    