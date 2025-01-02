import socket
import json

# AC:BC:32:D1:8F:C0

    
class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

    def connect(self):
        self.s.connect((self.host, self.port))
        data = self.s.recv(1024)
        assert data == b'ACK_CONN', 'error connecting!'

    def get_state(self):
        self.s.sendall(b'GET_STATE')
        data = self.s.recv(1024)
        data = json.loads(data.decode())
        assert data, 'Response Error!'
        return data

    def send_action(self, action):
        data = json.dumps(action)
        self.s.sendall(data.encode())
        data = self.s.recv(1024)
        data = json.loads(data.decode())
        assert data == 'ACK_ACTION', 'Response Error!'
        return True

    def close(self):
        self.s.close()

