import socket
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


HOST = "192.168.0.27"
PORT = 8000


class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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


if __name__ == "__main__":
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], '-')
    s = Client(HOST, PORT)

    def init():
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-1, 1)
        return ln,

    def update(frame):
        xdata.append(frame)
        data = s.send_data([1500 for _ in range(8)])
        ydata.append(data[0])
        ln.set_data(xdata, ydata)
        return ln,

    ani = FuncAnimation(
        fig,
        update,
        frames=np.linspace(0, 2*np.pi, 128),
        init_func=init,
        blit=True
    )
    plt.show()
    s.close()
