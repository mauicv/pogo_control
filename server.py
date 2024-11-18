import sys
import time
from server.channel import Channel
# from server.piggpio_servo_interface import PIGPIO_ServoInterface
from server.mpu6050_interface import MPU6050Interface


# SERVO_PINMAP = {1:4, 2:18, 3:27, 4:10, 5:20, 6:19, 7:13, 8:6}
HOST = "192.168.0.27"
POST = 8000


if __name__ == "__main__":
    mpu = MPU6050Interface()
    # servo = PIGPIO_ServoInterface(SERVO_PINMAP)
    # servo.update_angle([1500] * 8)

    def _handle_message(message):
    #     # servo.update_angle(message)
        data = mpu.get_data_sample()
        return data

    channel = Channel(host=HOST, port=POST)
    channel.serve(_handle_message)
