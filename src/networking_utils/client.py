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
        message_length = self.s.recv(4)
        message_length = int.from_bytes(message_length, byteorder='big')
        data = self._recv_all(self.s, message_length)
        data = json.loads(data.decode())
        assert data, 'Response Error!'
        return data

    def _recv_all(self, conn, n):
        """Helper function to receive exactly n bytes"""
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:  # Connection closed
                return None
            data.extend(packet)
        return data

    def close(self):
        self.s.close()
