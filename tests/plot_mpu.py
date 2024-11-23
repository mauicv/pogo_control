import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from client.client import Client

HOST = "192.168.0.27"
PORT = 8000

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
