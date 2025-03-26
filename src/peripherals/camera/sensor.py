from networking_utils.loop import Loop
from peripherals.camera.camera import Camera


class CameraSensor:
    def __init__(
            self,
            camera: Camera,
            **kwargs
        ):
        self.camera = camera
        self.buffer = []

    def capture(self):
        frame = self.camera.get_frame()
        self.buffer.append(frame)
        print(len(self.buffer))
        return frame.uuid

    def clean_buffer(self):
        self.buffer = []

    def deinit_camera_sensor(self):
        self.camera.close()
        self.clean_buffer()