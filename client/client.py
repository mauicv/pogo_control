import socket
import json

    
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

    def send_data(self, data):
        data = json.dumps(data)
        self.s.sendall(data.encode())
        data = self.s.recv(1024)
        print(data)
        data = json.loads(data.decode())
        assert data, 'Response Error!'
        return data

    def close(self):
        self.s.close()

