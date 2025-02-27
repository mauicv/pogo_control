from client.client import Client


class MultiClientInterface:
    def __init__(
            self,
            pogo_host: str,
            pogo_port: int,
            camera_host: str,
            camera_port: int
        ):
        self.pogo_client = Client(
            host=pogo_host,
            port=pogo_port
        )
        self.camera_client = Client(
            host=camera_host,
            port=camera_port
        )

    def connect(self):
        self.pogo_client.connect()
        self.camera_client.connect()

    def send_data(self, actions):
        servo_state, (ax, ay, az, gvx, gvy, gvz, pitch, roll), (overturned, last_mpus6050_sample_ts, last_servo_set_ts) \
            = self.pogo_client.send_data(actions)
        (position, distance, velocity, speed, yaw), (last_detection_ts,) \
            = self.camera_client.send_data({})
        world_state = [ax, ay, az, gvx, gvy, gvz, pitch, roll] + velocity + [speed] + [yaw]
        conditions = [overturned, last_mpus6050_sample_ts, last_servo_set_ts] + [last_detection_ts] + position + [distance]
        return (
            servo_state,
            world_state,
            conditions
        )

    def close(self):
        self.pogo_client.close()
        self.camera_client.close()
