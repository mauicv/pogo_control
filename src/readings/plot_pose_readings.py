import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from readings.data import PoseDataArray



def plot_pose_readings(client: Client):
    fig, axs = plt.subplots(ncols=2)
    init_xs = np.arange(100)
    init_ys = np.zeros(100)
    init_speeds = np.zeros(100)
    pose_data = PoseDataArray(
        xs=init_ys.tolist(),
        ys=init_ys.tolist(),
        zs=init_ys.tolist(),
        speeds=init_speeds.tolist(),
        avg_speeds=init_speeds.tolist(),
    )

    plot, = axs[0].plot(init_xs, init_ys, '-')

    axs[0].set_title("Location")
    axs[0].set_xlim(-100,100)
    axs[0].set_ylim(-100,100)

    speed_plot, = axs[1].plot(init_xs, init_speeds, '-')
    avg_speed_plot, = axs[1].plot(init_xs, init_speeds, '-')
    axs[1].set_title("Speed")
    axs[1].set_ylim(-50,50)

    def animate(i, client, pose_data: PoseDataArray):
        [data, _] = client.send_data({})
        tvec, rvec, velocity, speed, ts = data
        pose_data.update(tvec[0], speed)
        xs, ys, zs, speeds, avg_speeds = pose_data.get_data()
        plot.set_data(xs, ys)
        speed_plot.set_ydata(speeds)
        avg_speed_plot.set_ydata(avg_speeds)

    ani = animation.FuncAnimation(fig, animate, fargs=(client, pose_data, ), interval=25)
    plt.show()

