import cv2
import time
import numpy as np
from dataclasses import dataclass


@dataclass
class Frame:
    data: np.ndarray
    timestamp: float


class Camera:
    def __init__(
            self,
            height=480,
            width=640,
            fx=580.0,
            fy=580.0,
            cx=345.0,
            cy=244.0,
        ):
        self.height = height
        self.width = width
        self.timestamp = 0
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.open()

    def get_frame(self):
        if self.vc is None:
            return None
        if self.vc.isOpened():
            ret, frame = self.vc.read()
            if ret:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )
                frame = cv2.resize(
                    frame,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST
                )
                return Frame(
                    data=frame,
                    timestamp=time.time()
                )
            else:
                self.open()
                time.sleep(0.01)
                return self.get_frame()
        else:
            return None

    def open(self):
        try:
            self.vc = cv2.VideoCapture(0)
            self.vc.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V')
            )
            self.vc.set(
                cv2.CAP_PROP_FPS,
                30
            )
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.vc = None

    def deinit(self):
        try:
            if self.vc is not None:
                self.vc.release()
                self.vc = None
        except Exception as e:
            print(f"Error deinitializing camera: {e}")


if __name__ == "__main__":
    camera = Camera()
    frame = camera.get_frame()
    if frame is not None:
        cv2.imwrite("frame.jpg", frame.data)
        cv2.waitKey(1)
