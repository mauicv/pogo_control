import dotenv
dotenv.load_dotenv()

import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from dataclasses import dataclass, field


@dataclass
class AngularData:
    time: float = field(default_factory=time.time)
    adx: float = 0.0
    ady: float = 0.0
    adz: float = 0.0

    def update(self, data: list[float]):
        gyro_xs, gyro_ys, gyro_zs = data[3:6]
        dt = time.time() - self.time
        self.time = time.time()
        self.adx += gyro_xs * dt
        self.ady += gyro_ys * dt
        self.adz += gyro_zs * dt

    @property
    def rotation_vector(self):
        return np.array([self.adx, self.ady, self.adz])

    @property
    def normed(self):
        return self.rotation_vector / np.linalg.norm(self.rotation_vector)


def plot_orientation_3d(client: Client):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    origin = np.array([0, 0, 0])

    ax.legend()
    data = client.send_data({})
    angular_data = AngularData()
    angular_data.update(data)
    adx, ady, adz = angular_data.normed.tolist()
    ax.quiver(*origin, adx, ady, adz, color='b', length=1, normalize=False)

    def animate(i, client, angular_data: AngularData):
        data = client.send_data({})
        angular_data.update(data)
        adx, ady, adz = angular_data.normed.tolist()
        ax.cla()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.quiver(0, 0, 0, adx, ady, adz, color='b', length=1, normalize=False)


    ani = animation.FuncAnimation(fig, animate, fargs=(client, angular_data, ), interval=100)
    plt.show()
