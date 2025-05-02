import dotenv
dotenv.load_dotenv()
import numpy as np
from networking_utils.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from readings.data import PoseDataArray


def read_data(client):
    data = client.send_data({'command': 'read'})
    (x, y, distance, _, _, speed, yaw, _) = data
    return -x, y, distance, speed, yaw


def plot_pose_readings(client: Client):
    fig, axs = plt.subplot_mosaic(
        [
            ['location', 'distance'],
            ['yaw', 'speed']
        ],
        layout='constrained'
    )

    x, y, d, speed, yaw = read_data(client)

    init_xs = np.arange(100)
    init_pos_x = np.empty(100)
    init_pos_x.fill(x)
    init_pos_y = np.empty(100)
    init_pos_y.fill(y)
    init_speeds = np.empty(100)
    init_speeds.fill(speed)
    init_distances = np.empty(100)
    init_distances.fill(d)

    init_yaw = np.zeros((100))
    init_yaw.fill(yaw)

    pose_data = PoseDataArray(
        xs=init_pos_x.tolist(),
        ys=init_pos_y.tolist(),
        speeds=init_speeds.tolist(),
        distances=init_distances.tolist(),
        yaws=init_yaw.tolist()
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

    yaw_plot, = axs['yaw'].plot(init_xs, init_yaw, '-')
    axs['yaw'].set_title("yaw")
    axs['yaw'].set_ylim(0, 1)

    def animate(
            i,
            client,
            pose_data: PoseDataArray,
        ):
        x, y, d, speed, yaw = read_data(client)

        pose_data.update(
            x,
            y,
            speed,
            d,
            yaw
        )
        (
            xs,
            ys,
            speeds,
            distances,
            yaws,
        ) = pose_data.get_data()

        plot.set_data(xs, ys)
        speed_plot.set_ydata(speeds)
        distance_plot.set_ydata(distances)
        yaw_plot.set_ydata(yaws)

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

