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
    tvec, _, _, speed, ts = data
    d = np.linalg.norm(tvec)
    x, y, _ = tvec[0]
    return -x, y, d, speed


def plot_pose_readings(client: Client):
    fig, axs = plt.subplot_mosaic(
        [
            ['location', 'distance'],
            ['location', 'filtered_speed']
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
        avg_speeds=init_speeds.tolist(),
        distances=init_distances.tolist(),
        filtered_speeds=init_speeds.tolist(),
        filtered_distances=init_distances.tolist(),
    )


    plot, = axs['location'].plot(init_pos_x, init_pos_y, '-')
    filtered_plot, = axs['location'].plot(init_pos_x, init_pos_y, '-')
    axs['location'].set_title("Location")
    axs['location'].set_xlim(-40,40)
    axs['location'].set_ylim(-80,0)

    filtered_speed_plot, = axs['filtered_speed'].plot(init_xs, init_speeds, '-')
    axs['filtered_speed'].set_title("Speed")
    axs['filtered_speed'].set_ylim(-10, 10)

    distance_plot, = axs['distance'].plot(init_xs, init_distances, '-')
    filtered_distance_plot, = axs['distance'].plot(init_xs, init_distances, '-')
    axs['distance'].set_title("Distance")
    axs['distance'].set_ylim(0,100) 

    ds_filter = make_ds_filter(d, speed)
    xv_filter = make_xv_kalman_filter(x, y, 0, 0)

    def animate(
            i,
            client,
            pose_data: PoseDataArray,
        ):
        x, y, d, speed = extract_dv(client.send_data({}))

        ds_filter.predict()
        ds_filter.update(np.array([d]))

        xv_filter.predict()
        xv_filter.update(np.array([x, y]))

        pose_data.update(
            x,
            y,
            xv_filter.x[0],
            xv_filter.x[1],
            speed,
            d,
            ds_filter.x[1],
            ds_filter.x[0]
        )
        (
            xs,
            ys,
            filtered_xs,
            filtered_ys,
            speeds,
            avg_speeds,
            distances,
            filtered_speeds,
            filtered_distances
        ) = pose_data.get_data()

        plot.set_data(xs, ys)
        filtered_plot.set_data(filtered_xs, filtered_ys)

        filtered_speed_plot.set_ydata(filtered_speeds)

        distance_plot.set_ydata(distances)
        filtered_distance_plot.set_ydata(filtered_distances)

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

