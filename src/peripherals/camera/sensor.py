from peripherals.camera.camera import Camera
import os
from PIL import Image
from peripherals.camera.aruco_processor import ArucoSensorProcessor
from tqdm import tqdm
import cv2

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
            'read': self._read,
        }[command](**args)

    def _capture(self):
        frame = self.camera.get_frame()
        self.buffer.append(frame)
        return frame.uuid

    def _store(self, name='collection'):
        height=1536
        width=2048
        data = self.buffer
        os.makedirs('images', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(f'images/{name}.avi', fourcc, 5.0, (width, height), isColor=True)
        for frame in tqdm(data):
            colored_frame = cv2.cvtColor(frame.data, cv2.COLOR_GRAY2BGR)
            out.write(colored_frame)
        out.release()
        return name
    
    def _process(self):
        pose_data = []
        for frame in tqdm(self.buffer):
            self.aruco_processor.process(frame)
            data = self.aruco_processor.get_data()
            pose_data.append(data)
        return pose_data

    def _read(self):
        frame = self.camera.get_frame()
        self.aruco_processor.process(frame)
        data = self.aruco_processor.get_data()
        return data

    def _reset(self):
        self.buffer = []
        self.aruco_processor.init_variables()
        return True

    def deinit_camera_sensor(self):
        self.camera.close()
        self.reset()