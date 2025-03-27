from peripherals.camera.camera import Camera
import os
from PIL import Image
from peripherals.camera.aruco_processor import ArucoSensorProcessor

class CameraSensor:
    def __init__(
            self,
            camera: Camera,
            source_marker_id: int = 1,
            target_marker_id: int = 2,
            use_kalman_filter: bool = True,
        ):
        self.camera = camera
        self.buffer = []
        self.aruco_processor = ArucoSensorProcessor(
            source_marker_id=source_marker_id,
            target_marker_id=target_marker_id,
            use_kalman_filter=use_kalman_filter,
            camera=camera,
        )

    def _parse_command(self, message):
        command = message['command']
        args = message['args'] if 'args' in message else {}
        return command, args

    def handle_message(self, message):
        command, args = self._parse_command(message)
        return {
            'capture': self._capture,
            'store': self._store,
            'process': self._process,
            'reset': self._reset,
        }[command](**args)

    def _capture(self):
        frame = self.camera.get_frame()
        self.buffer.append(frame)
        return frame.uuid

    def _store(self, name='collection'):
        data = self.buffer
        os.makedirs('images', exist_ok=True)
        os.makedirs(f'images/{name}', exist_ok=True)
        for ind, frame in enumerate(data):
            Image.fromarray(frame.data).save(f'images/{name}/{ind:04d}_{frame.uuid}.png')
        return name
    
    def _process(self):
        data = []
        for frame in self.buffer:
            data.append(self.aruco_processor.process(frame))
        return data

    def _reset(self):
        self.buffer = []
        self.aruco_processor.init_variables()
        return True

    def deinit_camera_sensor(self):
        self.camera.close()
        self.reset()