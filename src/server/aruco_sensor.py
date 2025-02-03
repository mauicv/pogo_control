from filters.butterworth import _ButterworthFilter
from server.loop import Loop
from server.camera import Camera
import cv2
import numpy as np


class ArucoSensorMixin:
    def __init__(
            self,
            camera: Camera,
            aruco_sensor_update_interval: float = 0.01,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.markerSizeInCM = 10
        self.camera = camera
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.parameters
        )
        self._pos = 0.0
        self._pos_prev = 0.0
        self._vel = 0.0
        self._t_prev = 0.0

        self.v_filter = _ButterworthFilter(order=2, cutoff=2.0, fs=50.0)

        self.aruco_sensor_update_interval = max(0.05, aruco_sensor_update_interval)
        self.aruco_sensor_update_loop = Loop(
            interval=self.aruco_sensor_update_interval,
            func=self._compute_distance
        )
        self.aruco_sensor_update_loop.start()

    def _compute_distance(self):
        frame = self.camera.get_frame()
        if frame is None:
            return [0.0, 0.0]
        corners, ids, rejected = self.detector.detectMarkers(
            frame.data
        )
        if ids is not None:
            _ , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.markerSizeInCM,
                self.camera.camera_matrix,
                self.camera.dist_coeff,
            )
            self._pos = np.mean(tvec[:, :, 2], axis=0)[0]
            t_diff = self._t_prev - frame.timestamp
            self._vel = self.v_filter.filter(
                (self._pos - self._pos_prev) / t_diff
            )
            self._pos_prev = self._pos
            self._t_prev = frame.timestamp
        if ids is None:
            self._vel = 0.0

        return [self._vel]

    @property
    def aruco_velocity(self):
        return self._vel
    
    @property
    def aruco_position(self):
        return self._pos

    def deinit_aruco_sensor(self):
        """Clean up resources"""
        self.aruco_sensor_update_loop.stop()
        self.camera.close()
