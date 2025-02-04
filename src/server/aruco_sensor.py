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
        self.markerSizeInCM = 15
        self.camera = camera
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.parameters
        )
        self._distance = 0.0
        self._distance_prev = 0.0
        self._vel = 0.0
        self._t_prev = 0.0
        self._height = 0.0
        self._height_marker_detected = False
        self._velocity_marker_detected = False

        self.aruco_sensor_update_interval = max(0.05, aruco_sensor_update_interval)
        self.aruco_sensor_update_loop = Loop(
            interval=self.aruco_sensor_update_interval,
            func=self._compute_distance
        )
        self.aruco_sensor_update_loop.start()

    def _compute_height(self, ids, tvec):
        for aruco_id, id_tvec in zip(ids, tvec):
            if aruco_id == 2:
                self._height = -id_tvec[0, 1]
                self._height_marker_detected = True
        return None
    
    def _compute_distance_and_velocity(self, ids, tvec, ts):
        self._distance = np.mean(tvec[:, :, 2], axis=0)[0]
        t_diff = self._t_prev - ts
        self._vel = (self._distance - self._distance_prev) / t_diff
        self._distance_prev = self._distance
        self._t_prev = ts
        self._velocity_marker_detected = True
        return None

    def _compute_distance(self):
        frame = self.camera.get_frame()
        if frame is None:
            return
        corners, ids, rejected = self.detector.detectMarkers(
            frame.data
        )
        self._height_marker_detected = False
        self._velocity_marker_detected = False
        
        if ids is None: return
        ids = [id[0] for id in ids]
        
        # start testing code
        if 2 in ids:
            i = ids.index(2)
            corners = [corners[i]]
            ids = [ids[i]]
        # end testing code

        _ , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.markerSizeInCM,
            self.camera.camera_matrix,
            self.camera.dist_coeff,
        )

        self._compute_distance_and_velocity(ids, tvec, frame.timestamp)
        self._compute_height(ids, tvec)

    @property
    def aruco_velocity(self):
        return self._vel
    
    @property
    def aruco_distance(self):
        return self._distance
    
    @property
    def aruco_height(self):
        return self._height
    
    @property
    def aruco_height_marker_detected(self):
        return self._height_marker_detected
    
    @property
    def aruco_velocity_marker_detected(self):
        return self._velocity_marker_detected


    def deinit_aruco_sensor(self):
        """Clean up resources"""
        self.aruco_sensor_update_loop.stop()
        self.camera.close()
