from networking_utils.client import Client
from client.sample import Rollout
import torch
import uuid


class ClientInterface:
    def __init__(
            self,
            pogo_client: Client,
            camera_client: Client
        ):
        self.pogo_client = pogo_client
        self.camera_client = camera_client

    def connect(self):
        self.pogo_client.connect()
        self.camera_client.connect()

    def send_data(self, actions):
        servo_state, world_state, conditions = self.pogo_client.send_data({
            'command': 'act',
            'args': {
                'values': actions
            }
        })
        state = torch.tensor(servo_state + world_state)
        self.camera_client.send_data({'command': 'capture'})
        return (
            state,
            conditions
        )
    
    def set_servo_states(self, states: list[float]):
        self.pogo_client.send_data({
            'command': 'update_setpoint',
            'args': {
                'values': states
            }
        })
    
    def post_process(self, rollout: Rollout):
        data = self.camera_client.send_data({'command': 'process'})
        for ind, pose_data in enumerate(data):
            rollout.conditions[ind] = rollout.conditions[ind] + pose_data
        return rollout
    
    def save_images(self):
        name = str(uuid.uuid4())
        data = self.camera_client.send_data({
            'command': 'store',
            'args': {
                'name': name
            }
        })
        return name

    def reset(self):
        self.camera_client.send_data({'command': 'reset'})

    def close(self):
        self.pogo_client.close()
        self.camera_client.close()
