import dotenv
dotenv.load_dotenv()
import numpy as np
from networking_utils.client import Client
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from readings.data import SpeedDataArray


def get_speed(client: Client, pogo_client: Client):
    (vel, *_, ts) = client.send_data({'command': 'capture'})
    (s, *_) = pogo_client.send_data({})
    a = s[0]
    return vel, ts, a


def plot_pose_readings(client: Client, pogo_client: Client):
    fig, axs = plt.subplot_mosaic(
        [
            ['speed', 'detection_flags'],
            ['a', 'v']
        ],
        layout='constrained'
    )

    # fig, axs = plt.subplots(nrows=1, ncols=1)

    init_xs = np.arange(100)
    init_speed = np.empty(100)
    speed, _, a = get_speed(client, pogo_client)
    init_speed.fill(speed)
    init_detection_flags = np.empty(100)
    init_detection_flags.fill(0)
    init_pogo_a = np.empty(100)
    init_pogo_a.fill(a)
    init_pogo_v = np.empty(100)
    init_pogo_v.fill(0)

    pose_data = SpeedDataArray(
        speeds=init_speed.tolist(),
        detection_flags=init_detection_flags.tolist(),
        pogo_a=init_pogo_a.tolist(),
        pogo_v=init_pogo_v.tolist(),
    )

    speed_plot, = axs['speed'].plot(init_xs, init_speed, '-')
    axs['speed'].set_title("Speed")
    axs['speed'].set_ylim(-100, 100)

    detection_flags_plot, = axs['detection_flags'].plot(init_xs, init_detection_flags, '-')
    axs['detection_flags'].set_title("Detection Flags")
    axs['detection_flags'].set_ylim(-0.1, 1.1)

    pogo_a_plot, = axs['a'].plot(init_xs, init_pogo_a, '-')
    axs['a'].set_title("Pogo A")
    axs['a'].set_ylim(-0.1, 0.1)

    pogo_v_plot, = axs['v'].plot(init_xs, init_pogo_v, '-')
    axs['v'].set_title("Pogo V")
    axs['v'].set_ylim(-10, 10)

    def animate(
            i,
            client,
            pose_data: SpeedDataArray,
        ):
        speed, ts, a = get_speed(client, pogo_client)
        pose_data.update(speed, ts, a)
        speeds, detection_flags, pogo_a, pogo_v = pose_data.get_data()
        speed_plot.set_ydata(speeds)
        detection_flags_plot.set_ydata(detection_flags)
        pogo_a_plot.set_ydata(pogo_a)
        pogo_v_plot.set_ydata(pogo_v)

    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(
            client,
            pose_data,
        ),
        interval=25
    )
    plt.show()

