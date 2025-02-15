from filters.butterworth import _ButterworthFilter
from server.loop import Loop
from server.camera import Camera
import cv2
import numpy as np
import time


class ArucoSensorMixin:
    def __init__(
            self,
            camera: Camera,
            source_marker_id: int = 1,
            target_marker_id: int = 2,
            aruco_sensor_update_interval: float = 0.05,
            **kwargs
        ):
        self.markerSizeInCM = 4.5
        self.camera = camera
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.parameters
        )
        self._delta_tvec = np.array([0, 0, 0])
        self._delta_rvec = np.array([0, 0, 0])
        self._last_delta_tvec = np.array([0, 0, 0])
        self._last_delta_rvec = np.array([0, 0, 0])
        self._t_delta = 0
        self._velocity = np.array([0, 0, 0])
        self._speed = 0
        self._last_detection_ts = 0
        self.source_marker_id = source_marker_id
        self.target_marker_id = target_marker_id 

        self.aruco_sensor_update_interval = max(0.01, aruco_sensor_update_interval)
        self.aruco_sensor_update_loop = Loop(
            interval=self.aruco_sensor_update_interval,
            func=self._compute_distance
        )
        self.aruco_sensor_update_loop.start()

    def _compute_distance(self):
        start = time.time()
        frame = self.camera.get_frame()
        if frame is None: return

        corners, ids, rejected = self.detector.detectMarkers(
            frame.data
        )
        
        if ids is None: return
        ids = [id[0] for id in ids]
        if (self.source_marker_id not in ids) \
                or (self.target_marker_id not in ids):
            return

        source_index = ids.index(self.source_marker_id)
        target_index = ids.index(self.target_marker_id)

        rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.markerSizeInCM,
            self.camera.camera_matrix,
            self.camera.dist_coeff,
        )

        self._delta_tvec = tvec[target_index] - tvec[source_index]
        self._delta_rvec = rvec[target_index] - rvec[source_index]
        self._t_delta = frame.timestamp - self._last_detection_ts
        self._last_detection_ts = frame.timestamp
        diff = self._delta_tvec - self._last_delta_tvec
        self._velocity = diff / self._t_delta
        a = np.linalg.norm(self._delta_tvec)
        b = np.linalg.norm(self._last_delta_tvec)
        self._speed = (a - b) / self._t_delta
        self._last_delta_tvec = self._delta_tvec
        self._last_delta_rvec = self._delta_rvec
        end = time.time()
        # print(f"Pose computation time: {end - start}")

    def deinit_aruco_sensor(self):
        """Clean up resources"""
        self.aruco_sensor_update_loop.stop()
        self.camera.close()

    @property
    def delta_tvec(self):
        return self._delta_tvec.tolist()
    
    @property
    def delta_rvec(self):
        return self._delta_rvec.tolist()

    @property
    def last_detection_ts(self):
        return self._last_detection_ts

    @property
    def t_delta(self):
        return self._t_delta
    
    @property
    def velocity(self):
        return self._velocity.tolist()
    
    @property
    def speed(self):
        return self._speed
