import time


def move_robot(front_left_bottom, front_left_top, front_right_bottom, front_right_top, back_left_bottom, back_left_top, back_right_bottom, back_right_top):
    from peripherals.pogo import Pogo
    import pigpio
    from peripherals.pogo.mpu6050 import mpu6050

    gpio = pigpio.pi()
    mpu = mpu6050(0x68)
    pogo = Pogo(
        gpio=gpio,
        mpu=mpu,
        update_interval=0.01,
    )

    pogo.update_angle([
        front_right_top,
        front_right_bottom,
        front_left_top,
        front_left_bottom,
        back_right_top,
        back_right_bottom,
        back_left_top,
        back_left_bottom
    ])

    # pogo move-robot --back-right-top=0.2 --front-right-top=-0.3
    time.sleep(3)
    pogo.deinit()

