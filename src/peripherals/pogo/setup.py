from networking_utils.channel import Channel
import time


def setup_pogo_control(
    host: str,
    port: int,
    update_interval: float = 0.01,
):
    import pigpio
    from peripherals.pogo.pogo import Pogo
    from peripherals.pogo.mpu6050 import mpu6050

    gpio = pigpio.pi()
    mpu = mpu6050(0x68)
    pogo = Pogo(
        gpio=gpio,
        mpu=mpu,
        update_interval=update_interval,
    )

    # def _handle_message(message):
    #     pogo.update_angle(message)
    #     time.sleep(0.04)
    #     return pogo.get_data()

    channel = Channel(host=host, port=port)
    channel.serve(pogo._handle_message)


# def setup_pogo_sensor(
#     host: str,
#     port: int,
#     update_interval: float = 0.01,
# ):
#     from peripherals.pogo.mpu6050 import mpu6050
#     from peripherals.pogo.pogo import SensorPogo

#     mpu = mpu6050(0x68)
#     sensor_pogo = SensorPogo(mpu=mpu, update_interval=update_interval)

#     def _handle_message(message):   
#         data = sensor_pogo.get_data()
#         return data

#     channel = Channel(host=host, port=port)
#     channel.serve(_handle_message)