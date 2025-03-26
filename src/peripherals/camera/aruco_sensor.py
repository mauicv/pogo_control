from filters.kalman import KalmanDSFilter, KalmanXVFilter, KalmanYawFilter
from networking_utils.loop import Loop
from peripherals.camera.camera import Camera
import cv2
import numpy as np


class ArucoSensorMixin:
    def __init__(
            self,
            camera: Camera,
            source_marker_id: int = 1,
            target_marker_id: int = 2,
            aruco_sensor_update_interval: float = 0.05,
            use_kalman_filter: bool = True,
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
        self.use_kalman_filter = use_kalman_filter
        if self.use_kalman_filter:
            self.ds_filter = KalmanDSFilter(0)
            self.xv_filter = KalmanXVFilter(0, 0)
            self.yaw_filter = KalmanYawFilter(0)
        else:
            self._delta_tvec = np.array([0, 0, 0])
            self._delta_rvec = np.array([0, 0, 0])
            self._last_delta_tvec = np.array([0, 0, 0])
            self._last_delta_rvec = np.array([0, 0, 0])
            self._t_delta = 0
            self._velocity = np.array([0, 0, 0])
            self._speed = 0
            self._distance = 0
        self._yaw = 0
        self._last_detection_ts = 0
        self.source_marker_id = source_marker_id
        self.target_marker_id = target_marker_id
        self.aruco_sensor_update_interval = max(0.01, aruco_sensor_update_interval)
        self.aruco_sensor_update_loop = Loop(
            interval=self.aruco_sensor_update_interval,
            func=self._read
        )
        self.aruco_sensor_update_loop.start()

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

    def _read(self):
        # start = time.time()
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
        if self.use_kalman_filter:
            self._update_kalman_filter(frame, self._delta_tvec)
        else:
            self._update_raw(frame, self._delta_tvec)

        self._yaw = np.abs(rvec[target_index][0, 1])
        # end = time.time()
        # print(f"Time taken: {end - start}")

    def deinit_aruco_sensor(self):
        """Clean up resources"""
        self.aruco_sensor_update_loop.stop()
        self.camera.close()

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
