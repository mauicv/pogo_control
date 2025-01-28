from filters.butterworth import ButterworthFilter
from server.loop import Loop
from server.camera import Camera
import cv2
import numpy as np


class ArucoSensorMixin:
    def __init__(
            self,
            camera: Camera,
            aruco_sensor_update_interval: float = 0.01,
            filter: ButterworthFilter = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        # if not filter:
        #     self.filter = ButterworthFilter(num_components=3)
        # else:
        #     self.filter = filter
        self.markerSizeInCM = 10
        self.camera = camera
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.parameters
        )
        self._pos = 0.0
        self.aruco_sensor_update_loop = Loop(
            interval=aruco_sensor_update_interval,
            func=self._compute_distance
        )
        self.aruco_sensor_update_loop.start()

    def _compute_distance(self):
        frame = self.camera.get_frame()
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

    def get_pos(self):
        """Returns the most recent posistion data"""
        return [self._pos]

    def deinit_aruco_sensor(self):
        """Clean up resources"""
        self.aruco_sensor_update_loop.stop()
