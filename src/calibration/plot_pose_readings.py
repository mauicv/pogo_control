import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from calibration.data import PoseDataArray


def plot_pose_readings(client: Client):
    fig, axs = plt.subplots(ncols=2)
    pose_data = PoseDataArray()

    xs, ys, zs, speeds = pose_data.get_data()
    plot, = axs[0].plot(xs, ys, '-')

    axs[0].set_title("Location")
    axs[0].set_xlim(-100,100)
    axs[0].set_ylim(-100,100)

    axs[1].set_title("Speed")
    axs[1].set_xlim(0,100)

    def animate(i, client, pose_data: PoseDataArray):
        [data, _] = client.send_data({})
        tvec, rvec, velocity, speed, ts = data
        pose_data.update(tvec[0], speed)
        xs, ys, zs, speeds = pose_data.get_data()
        print(xs[-1], ys[-1])
        plot.set_data(xs, ys)
        axs[1].plot(speeds, '-')

    ani = animation.FuncAnimation(fig, animate, fargs=(client, pose_data, ), interval=25)
    plt.show()

