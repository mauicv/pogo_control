# Taken from https://docs.circuitpython.org/projects/pca9685/en/latest/examples.html#servo-example
import time
import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685


class ServoInterface:
    def __init__(self, channels):
        self.channels = channels
        self.i2c = board.I2C()
        self.pca = PCA9685(self.i2c)
        self.pca.frequency = 50
        self.servos = [ 
            servo.Servo(
                self.pca.channels[channel],
                min_pulse=500,
                max_pulse=2400
            )
            for channel in self.channels
        ]

    def update_angle(self, angles):
        for servo, angle in zip(self.servos, angles):
            servo.angle = angle

    def deinit(self):
        self.pca.deinit()
