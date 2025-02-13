from filters.butterworth import _ButterworthFilter
from server.loop import Loop
from server.camera import Camera
import cv2
import numpy as np


class ArucoSensorMixin:
    def __init__(
            self,
            camera: Camera,
            source_marker_id: int = 1,
            target_marker_id: int = 2,
            aruco_sensor_update_interval: float = 0.01,
            **kwargs
        ):
        self.markerSizeInCM = 15
        self.camera = camera
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.parameters
        )
        self._delta_tvec = None
        self._delta_rvec = None
        self._last_detection_ts = None
        self.source_marker_id = source_marker_id
        self.target_marker_id = target_marker_id 

        self.aruco_sensor_update_interval = max(0.05, aruco_sensor_update_interval)
        self.aruco_sensor_update_loop = Loop(
            interval=self.aruco_sensor_update_interval,
            func=self._compute_distance
        )
        self.aruco_sensor_update_loop.start()

    def _compute_distance(self):
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
        

        _ , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.markerSizeInCM,
            self.camera.camera_matrix,
            self.camera.dist_coeff,
        )

        self._delta_tvec = tvec[target_index] - tvec[source_index]
        self._delta_rvec = rvec[target_index] - rvec[source_index]
        self._last_detection_ts = frame.timestamp

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

