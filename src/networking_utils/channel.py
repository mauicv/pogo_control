import socket
import json
from json import JSONDecodeError


class Channel:
    def __init__(
            self,
            host,
            port
        ):
        self.host = host
        self.port = port

    def serve(self, function: callable):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            print(f"serving {function.__name__} at {self.host}/{self.port}")
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    conn.sendall(b"ACK_CONN")
                    print(f"Connected by {addr}")
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break

                        outputs = self._handle_request(data, function)
                        if outputs is None:
                            conn.sendall(b"ACK_MSG")
                            continue

                        conn.sendall(outputs)


    def _handle_request(self, data, function):
        try:
            data = json.loads(data.decode())
            output = function(data)
            output_bytes = json.dumps(output).encode()
            return output_bytes
        except JSONDecodeError as err:
            print(f"JSONDecodeError: {err}")
            return None