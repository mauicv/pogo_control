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
        data = json.loads(data.decode())
        assert data, 'Response Error!'
        return data

    def close(self):
        self.s.close()


class MultiClientInterface:
    def __init__(self):
        self.pogo_client = Client(host='192.168.1.100', port=5000)
        self.pogo_client.connect()
        self.camera_client = Client(host='192.168.1.100', port=5001)
        self.camera_client.connect()

    def send_data(self, data):
        pogo_data, camera_data = data
        pogo_data = self.pogo_client.send_data(pogo_data)
        camera_data = self.camera_client.send_data(camera_data)
        return pogo_data, camera_data

    def close(self):
        self.pogo_client.close()
        self.camera_client.close()
