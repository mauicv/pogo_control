import time
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from random import randint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.piggpio_async_servo_interface import PIGPIO_AsyncServoInterface
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


if __name__ == "__main__":
    SERVO_PINMAP = {0:4, 1:18, 2:27, 3:10, 4:20, 5:19, 6:13, 7:6}
    mock_pigpio = MockPIGPIO()
    async_servo_interface = PIGPIO_AsyncServoInterface(
        update_interval=0.01,
        pid_kp=0.1,
        pid_ki=0.01,
        pid_kd=0.001,
        pin_map=SERVO_PINMAP,
        pigpio=mock_pigpio
    )

    for action in rollout['actions']:
        async_servo_interface.update_angle(action)
        time.sleep(0.1)
    async_servo_interface.deinit()

    stretched_setpoints = np.repeat(
        np.array([async_servo_interface.normalized_action_to_servo_pwm(a[0]) for a in rollout['actions']]),
        np.ceil(len(mock_pigpio.values[4])/len(rollout['actions'])).astype(int)
    )

    plt.plot(mock_pigpio.values[4], label="Servo PID control")
    plt.plot(stretched_setpoints, label="Servo setpoint control")

    plt.legend()
    plt.show()