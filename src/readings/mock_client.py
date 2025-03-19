from client.client import Client
import random

class MockClient(Client):
    def __init__(self, host: str, port: int):
        pass

    def send_data(self, action: list[float]) -> list[float]:
        return [random.random() * 2 - 1 for _ in range(6 + 2+ 8)]
    
    def connect(self):
        return True
    
    def close(self):
        pass
