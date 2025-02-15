import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from readings.data import PoseDataArray
from filterpy.kalman import KalmanFilter


def extract_dv(data):
    [data, _] = data
    tvec, _, _, speed, ts = data
    d = np.linalg.norm(tvec)
    return tvec[0], d, speed

def plot_pose_readings(client: Client):
    fig, axs = plt.subplots(ncols=3)
    tvec, d, speed = extract_dv(client.send_data({}))
    init_xs = np.arange(100)
    init_pos_x = np.empty(100)
    init_pos_x.fill(tvec[0])
    init_pos_y = np.empty(100)
    init_pos_y.fill(tvec[1])
    init_speeds = np.empty(100)
    init_speeds.fill(speed)
    init_distances = np.empty(100)
    init_distances.fill(d)

    pose_data = PoseDataArray(
        xs=init_pos_x.tolist(),
        ys=init_pos_y.tolist(),
        speeds=init_speeds.tolist(),
        avg_speeds=init_speeds.tolist(),
        distances=init_distances.tolist(),
    )

    plot, = axs[0].plot(init_pos_x, init_pos_y, '-')

    axs[0].set_title("Location")
    axs[0].set_xlim(-50,50)
    axs[0].set_ylim(-80,0)

    speed_plot, = axs[1].plot(init_xs, init_speeds, '-')
    avg_speed_plot, = axs[1].plot(init_xs, init_speeds, '-')
    axs[1].set_title("Speed")
    axs[1].set_ylim(-50,50)

    distance_plot, = axs[2].plot(init_xs, init_distances, '-')
    axs[2].set_title("Distance")
    axs[2].set_ylim(0,100) 

    # KalmanFilter(dim_x=2, dim_z=1)

    def animate(i, client, pose_data: PoseDataArray):
        tvec, d, speed = extract_dv(client.send_data({}))
        print(tvec, d, speed)
        pose_data.update(tvec, speed, d)
        xs, ys, zs, speeds, avg_speeds, distances = pose_data.get_data()
        plot.set_data(xs, ys)
        speed_plot.set_ydata(speeds)
        avg_speed_plot.set_ydata(avg_speeds)
        distance_plot.set_ydata(distances)

    ani = animation.FuncAnimation(fig, animate, fargs=(client, pose_data, ), interval=90)
    plt.show()

