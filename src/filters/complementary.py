import math
import time
import numpy as np
GRAVITY = 9.80665


class ComplementaryFilter:
    def __init__(self, alpha=0.95):
        self.rollG = 0
        self.pitchG = 0
        self.rollComp = 0
        self.pitchComp = 0
        self.a_roll = 0
        self.a_pitch = 0
        self.t_last = None
        self.alpha = alpha

    def update(self, acc_data, gyro_data):
        if self.t_last is None:
            self.t_last = time.process_time()
            return

        x_accel, y_accel, z_accel = acc_data
        x_gyro, y_gyro, _ = gyro_data

        self.a_roll = math.atan2(-x_accel, z_accel)*180/math.pi
        self.a_pitch = math.atan2(y_accel, z_accel)*180/math.pi

        t_start = time.process_time()        
        dt = (t_start - self.t_last)
        self.rollComp = self.a_roll * (1 - self.alpha) \
            + self.alpha * (self.rollComp + y_gyro * dt)
        self.pitchComp = self.a_pitch * (1 - self.alpha) \
             + self.alpha * (self.pitchComp + x_gyro * dt)
        
        self.t_last = t_start

    @property
    def roll(self):
        return self.rollComp / 180

    @property
    def pitch(self):
        return self.pitchComp / 180
    
    @property
    def overturned(self):
        return self.roll > 0.4 or self.roll < -0.4
        
    @property
    def g_x(self):
        return GRAVITY*math.cos(self.roll*math.pi/180)

    @property
    def g_y(self):
        return GRAVITY*math.sin(self.pitch*math.pi/180)
    
    @property
    def g_xy(self):
        return self.g_x, self.g_y
