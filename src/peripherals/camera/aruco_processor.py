from filters.kalman import KalmanDSFilter, KalmanXVFilter, KalmanYawFilter
from peripherals.camera.camera import Frame, Camera
import cv2
import numpy as np


class ArucoSensorProcessor:
    def __init__(
            self,
            source_marker_id: int = 1,
            target_marker_id: int = 2,
            use_kalman_filter: bool = True,
            camera: Camera = None,
        ):
        self.markerSizeInCM = 4.5
        self.camera = camera
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.source_marker_id = source_marker_id
        self.target_marker_id = target_marker_id
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.parameters
        )
        self.use_kalman_filter = use_kalman_filter
        self.init_variables()

    def init_variables(self):
        if self.use_kalman_filter:
            self.ds_filter = KalmanDSFilter(0.0)
            self.xv_filter = KalmanXVFilter(0.0, 0.0)
            self.yaw_filter = KalmanYawFilter(0.0)
        self._delta_tvec = np.array([0.0, 0.0, 0.0])
        self._delta_rvec = np.array([0.0, 0.0, 0.0])
        self._last_delta_tvec = np.array([0.0, 0.0, 0.0])
        self._last_delta_rvec = np.array([0.0, 0.0, 0.0])
        self._t_delta = 0.0
        self._velocity = np.array([0.0, 0.0, 0.0])
        self._speed = 0.0
        self._distance = 0.0
        self._yaw = 0.0
        self._last_detection_ts = 0

    def _update_kalman_filter(self, frame, delta_tvec):
        x, y, _ = delta_tvec[0]
        distance = np.linalg.norm(delta_tvec)
        self._last_detection_ts = frame.timestamp
        self.ds_filter(distance)
        self.xv_filter(x, y)
        self.yaw_filter(self._yaw)

    def _update_raw(self, frame, delta_tvec):
        self._delta_tvec = delta_tvec
        self._t_delta = frame.timestamp - self._last_detection_ts
        self._last_detection_ts = frame.timestamp
        diff = self._delta_tvec - self._last_delta_tvec
        self._velocity = diff / self._t_delta
        a = np.linalg.norm(self._delta_tvec)
        b = np.linalg.norm(self._last_delta_tvec)
        self._speed = (a - b) / self._t_delta
        self._last_delta_tvec = self._delta_tvec
        self._distance = a

    def process(self, frame: Frame):
        corners, ids, _ = self.detector.detectMarkers(
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
        if self.use_kalman_filter:
            self._update_kalman_filter(frame, self._delta_tvec)
        else:
            self._update_raw(frame, self._delta_tvec)

        self._yaw = np.abs(rvec[target_index][0, 1])

    def get_data(self):
        return [
            self.position[0],
            self.position[1],
            self.distance,
            self.velocity[0],
            self.velocity[1],
            self.speed,
            self.yaw,
            self.last_detection_ts,
        ]

    @property
    def position(self):
        if self.use_kalman_filter:
            return [self.xv_filter.x[0], self.xv_filter.x[1]]
        else:
            return self._delta_tvec.tolist()
    
    @property
    def distance(self):
        if self.use_kalman_filter:
            return self.ds_filter.x[0]
        else:
            return self._distance

    @property
    def last_detection_ts(self):
        return self._last_detection_ts

    @property
    def t_delta(self):
        return self._t_delta
    
    @property
    def velocity(self):
        if self.use_kalman_filter:
            return self.xv_filter.x[2:].tolist()
        else:
            return self._velocity.tolist()
    
    @property
    def speed(self):
        if self.use_kalman_filter:
            return self.ds_filter.x[1]
        else:
            return self._speed
        
    @property
    def yaw(self):
        if self.use_kalman_filter:
            return self.yaw_filter.x[0]/np.pi
        else:
            return self._yaw/np.pi
