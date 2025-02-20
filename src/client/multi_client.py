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
        self.pogo_client.connect()
        self.camera_client = Client(
            host=camera_host,
            port=camera_port
        )
        self.camera_client.connect()

    def send_data(self, actions):
        servo_state, world_state, pogo_conditions \
            = self.pogo_client.send_data(actions)
        camera_state, camera_conditions \
            = self.camera_client.send_data({})
        return (
            servo_state,
            world_state + camera_state,
            pogo_conditions + camera_conditions
        )

    def close(self):
        self.pogo_client.close()
        self.camera_client.close()
