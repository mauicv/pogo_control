import cv2
import time
import numpy as np
from dataclasses import dataclass


import numpy as np

c_params = {
    'camera_matrix': np.array([
        [581.5776347427477, 0.0, 345.120325167835],
        [0.0, 580.6905120295608, 244.63647666611948],
        [0.0, 0.0, 1.0]
    ]),
    'dist_coeff': np.array([[
        -0.2634978333836847,
        -0.540562680385177,
        -0.00021971548190154595,
        -0.0029920783484676796,
        1.835011346570344
    ]])
}


@dataclass
class Frame:
    data: np.ndarray
    timestamp: float


class Camera:
    def __init__(
            self,
            input_source=-1,
            camera_matrix=c_params['camera_matrix'],
            dist_coeff=c_params['dist_coeff'],
            height=480,
            width=640,
            fx=580.0,
            fy=580.0,
            cx=345.0,
            cy=244.0,
        ):
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.input_source = input_source
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff
        self.open(input_source=input_source)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeff,
            (self.width, self.height),
            1,
            (self.width, self.height)
        )
        self.newcameramtx = newcameramtx
        self.roi = roi

    def get_frame(self):
        if self.vc is None:
            return None
        if self.vc.isOpened():
            ret, frame = self.vc.read()
            if ret:
                frame = cv2.undistort(
                    frame,
                    c_params['camera_matrix'],
                    c_params['dist_coeff'],
                    None,
                    self.newcameramtx
                )
                x, y, w, h = self.roi
                frame = frame[y:y+h, x:x+w]
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )
                return Frame(
                    data=frame,
                    timestamp=time.time()
                )
            else:
                time.sleep(0.1)
                self.open()
                return self.get_frame()
        else:
            return None

    def open(self, input_source):
        self.close()
        try:
            self.vc = cv2.VideoCapture(input_source)
            self.vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.vc.set(cv2.CAP_PROP_FPS, 30)
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.vc = None

    def close(self):
        try:
            if self.vc is not None:
                self.vc.release()
                self.vc = None
        except Exception as e:
            print(f"Error deinitializing camera: {e}")
