from peripherals.camera_sensor.aruco_sensor import ArucoSensorMixin


class PoseSensor(ArucoSensorMixin):
    def __init__(
            self,
            update_interval: float = 0.01,
            camera=None,
        ):

        if not camera:
            from camera_sensor.camera import Picamera2Camera as Camera
            camera = Camera(input_source="main")

        super().__init__(
            aruco_update_interval=update_interval,
            camera=camera
        )

    def get_data(self):
        state_data = [
            self.position,
            self.distance,
            self.velocity,
            self.speed,
            self.yaw,
        ]
        conditions_data = [
            self.last_detection_ts,
        ]
        return [state_data, conditions_data]

    def deinit(self):
        self.deinit_aruco_sensor()
