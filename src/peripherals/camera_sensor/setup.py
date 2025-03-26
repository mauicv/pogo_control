import json
import pprint
import numpy as np


def setup_camera_sensor(
    host: str,
    port: int,
    camera_matrix_file: str,
    update_interval: float = 0.01,
):
    from networking_utils.channel import Channel
    from peripherals.camera_sensor.pose_sensor import PoseSensor
    from peripherals.camera_sensor.camera import Picamera2Camera as Camera

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
    pose_sensor = PoseSensor(
        camera=camera,
        update_interval=update_interval
    )

    def _handle_message(message):
        # time.sleep(0.08)
        data = pose_sensor.get_data()
        return data

    channel = Channel(host=host, port=port)
    channel.serve(_handle_message)
