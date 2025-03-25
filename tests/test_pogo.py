import time
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from peripherals.pogo import Pogo
from rollout import rollout


class MockPIGPIO:
    def __init__(self):
        self.connected = True
        self.values = {
            4: [],
            18: [],
            27: [],
            10: [],
            20: [],
            19: [],
            13: [],
            6: [],
        }

    def get_servo_pulsewidth(self, pin):
        return 1900

    def set_servo_pulsewidth(self, pin, value):
        self.values[pin].append(value)

    def stop(self):
        pass


class MockValues:
    def __init__(self, values):
        self._values = values

    def values(self):
        return self._values


class Mock_mpu6050:
    def __init__(self):
        self.data = [0] * 6
        self.last_update = time.time()
        self.intervals = []

    def get_accel_data(self):
        time.sleep(0.001)
        self.intervals.append(time.time() - self.last_update)
        self.last_update = time.time()
        self.data = MockValues([(random.uniform(-10, 10)) for _ in range(3)])
        return self.data

    def get_gyro_data(self):
        time.sleep(0.001)
        self.data = MockValues([(random.uniform(-10, 10)) for _ in range(3)])
        return self.data


if __name__ == "__main__":
    mock_pigpio = MockPIGPIO()
    mock_mpu = Mock_mpu6050()
    pogo = Pogo(
        update_interval=0.01,
        gpio=mock_pigpio,
        mpu=mock_mpu,
    )

    for action in tqdm(rollout['actions']):
        pogo.update_angle(action)
        time.sleep(0.1)
    pogo.deinit()

    SERVO_PWM_THRESHOLD_MIN: int = 500
    SERVO_PWM_THRESHOLD_MAX: int = 2500
    HALF_RANGE = (SERVO_PWM_THRESHOLD_MAX - SERVO_PWM_THRESHOLD_MIN) / 2
    to_pwm = lambda value: int(SERVO_PWM_THRESHOLD_MIN + (1 + value) * HALF_RANGE)

    stretched_setpoints = np.repeat(
        np.array([to_pwm(a[0]) for a in rollout['actions']]),
        np.ceil(len(mock_pigpio.values[4])/len(rollout['actions'])).astype(int)
    )

    plt.plot(mock_pigpio.values[4], label="Servo PID control")
    plt.plot(stretched_setpoints, label="Servo setpoint control")

    plt.legend()
    plt.show()