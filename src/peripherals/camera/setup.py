import json
import pprint
import numpy as np


def setup_camera_sensor(
    host: str,
    port: int,
    camera_matrix_file: str,
    live: bool = False,
    use_kalman_filter: bool = False,
):
    from networking_utils.channel import Channel
    from peripherals.camera.camera import Picamera2Camera as Camera
    from peripherals.camera.sensor import CameraSensor, LiveCameraSensor

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

    if live:
        camera_sensor = LiveCameraSensor(
            camera=camera,
            use_kalman_filter=use_kalman_filter,
        )
    else:
        camera_sensor = CameraSensor(
            camera=camera,
            use_kalman_filter=use_kalman_filter,
        )

    channel = Channel(host=host, port=port)
    channel.serve(camera_sensor.handle_message)
