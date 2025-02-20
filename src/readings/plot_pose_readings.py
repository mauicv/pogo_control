import dotenv
dotenv.load_dotenv()
import numpy as np
from client.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from readings.data import PoseDataArray
from filters.kalman import make_ds_filter, make_xv_kalman_filter


def extract_dv(data):
    [data, _] = data
    position, distance,  _, speed = data
    x, y = position
    return -x, y, distance, speed


def plot_pose_readings(client: Client):
    fig, axs = plt.subplot_mosaic(
        [
            ['location', 'distance'],
            ['location', 'speed']
        ],
        layout='constrained'
    )

    x, y, d, speed = extract_dv(client.send_data({}))
    init_xs = np.arange(100)
    init_pos_x = np.empty(100)
    init_pos_x.fill(x)
    init_pos_y = np.empty(100)
    init_pos_y.fill(y)
    init_speeds = np.empty(100)
    init_speeds.fill(speed)
    init_distances = np.empty(100)
    init_distances.fill(d)

    pose_data = PoseDataArray(
        xs=init_pos_x.tolist(),
        ys=init_pos_y.tolist(),
        speeds=init_speeds.tolist(),
        distances=init_distances.tolist(),
    )

    plot, = axs['location'].plot(init_pos_x, init_pos_y, '-')
    axs['location'].set_title("Location")
    axs['location'].set_xlim(-40,40)
    axs['location'].set_ylim(-80,0)

    speed_plot, = axs['speed'].plot(init_xs, init_speeds, '-')
    axs['speed'].set_title("Speed")
    axs['speed'].set_ylim(-10, 10)

    distance_plot, = axs['distance'].plot(init_xs, init_distances, '-')
    axs['distance'].set_title("Distance")
    axs['distance'].set_ylim(0,100)

    def animate(
            i,
            client,
            pose_data: PoseDataArray,
        ):
        x, y, d, speed = extract_dv(client.send_data({}))
        print(speed)

        pose_data.update(
            x,
            y,
            speed,
            d,
        )
        (
            xs,
            ys,
            speeds,
            distances,
        ) = pose_data.get_data()

        plot.set_data(xs, ys)
        speed_plot.set_ydata(speeds)
        distance_plot.set_ydata(distances)

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            client,
            pose_data,
        ),
        interval=250
    )
    plt.show()

