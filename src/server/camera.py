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
        self.input_source = input_source
        self.open()

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
        if self.vc is None or not self.vc.isOpened():
            print("Camera not initialized or opened. Attempting to reopen...")
            self.open()
            if self.vc is None or not self.vc.isOpened():
                return None
            
        ret, frame = self.vc.read()
        if not ret:
            print("Failed to read frame. Attempting to reopen camera...")
            time.sleep(0.1)
            self.open()
            return self.get_frame()
        
        frame = cv2.undistort(
            frame,
            self.camera_matrix,
            self.dist_coeff,
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

    def open(self):
        self.close()
        try:
            # Check if camera device exists and is accessible
            self.vc = cv2.VideoCapture(self.input_source)
            if not self.vc.isOpened():
                print(f"Failed to open camera device at input_source={self.input_source}")
                self.vc = None
                return

            # Try to set camera properties
            success_fourcc = self.vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            success_fps = self.vc.set(cv2.CAP_PROP_FPS, 30)
            
            if not (success_fourcc and success_fps):
                print("Warning: Failed to set some camera properties")
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.vc = None

    def close(self):
        try:
            if self.vc is not None:
                self.vc.release()
                cv2.destroyAllWindows()
                self.vc = None
        except Exception as e:
            print(f"Error deinitializing camera: {e}")
