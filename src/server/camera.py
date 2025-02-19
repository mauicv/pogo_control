import cv2
import time
import numpy as np
from dataclasses import dataclass
from picamera2 import Picamera2
from libcamera import controls
import numpy as np
import pprint
import time

@dataclass
class Frame:
    data: np.ndarray
    timestamp: float


class Camera:
    def __init__(
            self,
            camera_matrix,
            dist_coeff,
            height=480,
            width=640,
            undistort=False,
        ):
        self.undistort = undistort
        self.height = height
        self.width = width
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeff,
            (self.width, self.height),
            1,
            (self.width, self.height)
        )
        self.newcameramtx = newcameramtx
        self.roi = roi

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeff,
            None,
            self.newcameramtx,
            (self.width, self.height),
            5
        )

    def get_frame(self):
        frame = self.capture()
        if frame is None:
            return None
        if self.undistort:
            frame = cv2.remap(
                frame,
                self.mapx,
                self.mapy,
                cv2.INTER_LINEAR,
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

    def capture(self):
        raise NotImplementedError("Capture method not implemented")

    def open(self):
        raise NotImplementedError("Open method not implemented")

    def close(self):
        raise NotImplementedError("Close method not implemented")


class Cv2Camera(Camera):
    def __init__(
            self,
            camera_matrix,
            dist_coeff,
            input_source=-1,
            height=480,
            width=640,
        ):
        super().__init__(
            camera_matrix,
            dist_coeff,
            height,
            width,
        )
        self.input_source = input_source
        self.vc = None
        self.open()

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

    def capture(self):
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
            return self.capture()

    def close(self):
        try:
            if self.vc is not None:
                self.vc.release()
                cv2.destroyAllWindows()
                self.vc = None
        except Exception as e:
            print(f"Error deinitializing camera: {e}")


class Picamera2Camera(Camera):
    def __init__(
            self,
            camera_matrix,
            dist_coeff,
            input_source="main",
            height=480,
            width=640,
        ):
        super().__init__(
            camera_matrix,
            dist_coeff,
            height,
            width,
        )
        self.input_source = input_source
        self.vc = Picamera2()
        config = self.vc.create_video_configuration(
            main={"size": (width, height)},
            controls={"AfMode": controls.AfModeEnum.Continuous}
        )
        self.vc.configure(config)
        self.vc.set_controls({
            "FrameRate": 40,
        })
        pprint.pprint(self.vc.controls)
        self.open()

    def open(self):
        self.vc.start()

    def capture(self):
        try:
            frame = self.vc.capture_array(self.input_source)
            return frame
        except Exception as e:
            return None

    def close(self):
        self.vc.close()