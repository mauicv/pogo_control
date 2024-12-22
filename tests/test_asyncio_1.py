import time
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.piggpio_async_servo_interface import PIGPIO_AsyncServoInterface, Loop


class MockPIGPIO:
    def __init__(self):
        self.connected = True
        self.values = {18: []}

    def get_servo_pulsewidth(self, pin):
        return 0
    
    def set_servo_pulsewidth(self, pin, value):
        self.values[pin].append(value)
        print(f"Setting servo pulsewidth for pin {pin} to {value}")

    def stop(self):
        pass


if __name__ == "__main__":
    pin_map = {
        0: 18,
    }
    mock_pigpio = MockPIGPIO()
    async_servo_interface = PIGPIO_AsyncServoInterface(
        update_interval=0.01,
        pid_kp=0.1,
        pid_ki=0.01,
        pid_kd=0.001,
        pin_map=pin_map,
        pigpio=mock_pigpio
    )

    async_servo_interface.normalized_action_to_servo_pwm = lambda x: x

    async_servo_interface.update_angle_setpoints([0.5])
    time.sleep(2)
    async_servo_interface.update_angle_setpoints([-0.5])
    time.sleep(2)
    async_servo_interface.update_angle_setpoints([0.5])
    time.sleep(2)
    async_servo_interface.update_angle_setpoints([0.6])
    time.sleep(2)
    async_servo_interface.deinit()

    setpoints = np.array([[0.5], [-0.5], [0.5], [0.6]])
    stretched_setpoints = np.repeat(
        setpoints,
        np.ceil(len(mock_pigpio.values[18])/len(setpoints)).astype(int)
    )

    plt.plot(mock_pigpio.values[18], label="Servo 0")
    plt.plot(stretched_setpoints, label="Servo 0 setpoint control")
    plt.legend()
    plt.show()