import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from calibration.data import PoseDataArray


def plot_pose_readings(client: Client):
    fig, axs = plt.subplots()
    sensor_data = PoseDataArray()

    xs, ys, zs = sensor_data.get_data()
    plot, = axs.plot(xs, ys, '-')

    axs.set_title("Location")
    plt.xlim(-100,100)
    plt.ylim(-100,100)

    def animate(i, client, pose_data: PoseDataArray):
        [data, _] = client.send_data({})
        tvec, rvec, ts = data
        pose_data.update(tvec[0])
        xs, ys, zs = pose_data.get_data()
        print(xs[-1], ys[-1])
        plot.set_data(xs, ys)

    ani = animation.FuncAnimation(fig, animate, fargs=(client, sensor_data, ), interval=25)
    plt.show()

