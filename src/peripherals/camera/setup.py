import json
import pprint
import numpy as np


def setup_camera_sensor(
    host: str,
    port: int,
    camera_matrix_file: str,
):
    from networking_utils.channel import Channel
    from peripherals.camera.camera import Picamera2Camera as Camera
    from peripherals.camera.sensor import CameraSensor

    with open(camera_matrix_file, 'r') as f:
        c_params = json.load(f)
        pprint.pprint(c_params)

    camera_matrix = np.array(c_params['camera_matrix'])
    dist_coeff = np.array(c_params['dist_coeff'])

    camera = Camera(
        input_source="main",
        height=1536,
        width=2048,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
    )
    camera_sensor = CameraSensor(
        camera=camera,
    )

    def _handle_message(message):
        data = camera_sensor.capture()
        return data

    channel = Channel(host=host, port=port)
    channel.serve(_handle_message)
