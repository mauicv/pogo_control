from networking_utils.client import Client
from client.sample import Rollout
import torch


class StandingClientInterface:
    def __init__(
            self,
            pogo_client: Client,
        ):
        self.pogo_client = pogo_client

    def connect(self):
        self.pogo_client.connect()

    def send_data(self, actions):
        servo_state, world_state, conditions = self.pogo_client.send_data(actions)
        state = torch.tensor(servo_state + world_state)
        return state, conditions
    
    def post_process(self, rollout: Rollout):
        return rollout
    
    def reset(self):
        pass

    def close(self):
        self.pogo_client.close()


class WalkingClientInterface:
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
        servo_state, world_state, conditions = self.pogo_client.send_data(actions)
        state = torch.tensor(servo_state + world_state)
        self.camera_client.send_data({'command': 'capture'})
        return (
            state,
            conditions
        )
    
    def post_process(self, rollout: Rollout):
        data = self.camera_client.send_data({'command': 'process'})
        for ind, pose_data in enumerate(data):
            rollout.conditions[ind] = rollout.conditions[ind] + pose_data
        return rollout

    def reset(self):
        self.camera_client.send_data({'command': 'reset'})

    def close(self):
        self.pogo_client.close()
        self.camera_client.close()
