from filters.kalman import KalmanDSFilter, KalmanXYZFilter, KalmanYawFilter
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
            # self.front_xyx_filter = KalmanXYZFilter(0.0, 0.0, 0.0)
            # self.back_xyz_filter = KalmanXYZFilter(0.0, 0.0, 0.0)
            self.front_xyx_filter = None
            self.back_xyz_filter = None

        self._last_front_tvec = None
        self._last_back_tvec = None
        # self._last_tvec = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self._orientation = None
        # self._orientation = np.array([0.0, 0.0, 0.0])

        self._t_delta = 0.0
        self._last_detection_ts = 0

    def _update_kalman_filter(self, frame, tvec, source_index, target_index):
        self._t_delta = frame.timestamp - self._last_detection_ts
        self._last_detection_ts = frame.timestamp
        self._orientation = tvec[target_index] - tvec[source_index]

        if self._last_front_tvec is None:
            self._last_front_tvec = tvec[source_index]
        if self._last_back_tvec is None:
            self._last_back_tvec = tvec[target_index]
        if self.front_xyx_filter is None:
            ((x, y, z), ) = tvec[source_index]
            self.front_xyx_filter = KalmanXYZFilter(x, y, z)
        if self.back_xyz_filter is None:
            ((x, y, z), ) = tvec[target_index]
            self.back_xyz_filter = KalmanXYZFilter(x, y, z)

        front_tvec = tvec[source_index][0]
        front_tvec = np.array(self.front_xyx_filter(*front_tvec))
        back_tvec = tvec[target_index][0]
        back_tvec = np.array(self.back_xyz_filter(*back_tvec))

        front_diff = front_tvec - self._last_front_tvec
        back_diff = back_tvec - self._last_back_tvec
        
        self._last_front_tvec = front_tvec
        self._last_back_tvec = back_tvec
        self._last_detection_ts = frame.timestamp
        vel_1 = np.dot(np.array(front_diff), self._orientation[0]).item()
        vel_2 = np.dot(np.array(back_diff), self._orientation[0]).item()
        self._speed = (vel_1 + vel_2) / 2 * self._t_delta

    def _update_raw(self, frame, tvec, source_index, target_index):
        if self._last_front_tvec is None:
            self._last_front_tvec = tvec[source_index]
        if self._last_back_tvec is None:
            self._last_back_tvec = tvec[target_index]

        self._t_delta = frame.timestamp - self._last_detection_ts
        self._last_detection_ts = frame.timestamp
        self._orientation = tvec[target_index] - tvec[source_index]
        front_diff = tvec[source_index] - self._last_front_tvec
        back_diff = tvec[target_index] - self._last_back_tvec
        self._last_front_tvec = tvec[source_index]
        self._last_back_tvec = tvec[target_index]
        vel_1 = np.dot(front_diff[0], self._orientation[0])
        vel_2 = np.dot(back_diff[0], self._orientation[0])
        self._speed = (vel_1 + vel_2) / 2 * self._t_delta

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

        _ , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.markerSizeInCM,
            self.camera.camera_matrix,
            self.camera.dist_coeff,
        )

        if self.use_kalman_filter:
            self._update_kalman_filter(frame, tvec, source_index, target_index)
        else:
            self._update_raw(frame, tvec, source_index, target_index)

    def get_data(self):
        return [
            self._speed,
            *self._orientation[0],
            self._last_detection_ts,
        ]