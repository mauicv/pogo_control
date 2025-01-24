import math
import time
import numpy as np
GRAVITY = 9.80665


class ComplementaryFilter:
    def __init__(self):
        self.rollG = 0
        self.pitchG = 0
        self.rollComp = 0
        self.pitchComp = 0
        self.a_roll = 0
        self.a_pitch = 0
        self.t_last = None

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
        self.rollComp= self.a_roll*.05 + .95*(self.rollComp+y_gyro*dt)
        self.pitchComp= self.a_pitch*.05 + .95*(self.pitchComp+x_gyro*dt)
        
        self.t_last = t_start

    @property
    def roll(self):
        return self.rollComp

    @property
    def pitch(self):
        return self.pitchComp
    
    @property
    def g_x(self):
        return GRAVITY*math.cos(self.roll*math.pi/180)

    @property
    def g_y(self):
        return GRAVITY*math.sin(self.pitch*math.pi/180)
    
    @property
    def g_xy(self):
        return self.g_x, self.g_y
        
    
class SimpleVelocityFilter:
    def __init__(self, alpha=0.9):
        self.t_last = None
        self.a_x = 0
        self.a_y = 0
        self.v_x = 0
        self.v_y = 0
        self.alpha = alpha

    def update(self, acc_data, g_xy):
        if self.t_last is None:
            self.t_last = time.process_time()
            return

        x_accel, y_accel, _ = acc_data
        g_x, g_y = g_xy
        t_start = time.process_time()        
        dt = (t_start - self.t_last)
        self.a_x = x_accel - g_x
        self.a_y = y_accel - g_y
        self.v_x = self.alpha * (self.v_x + self.a_x*dt)
        self.v_y = self.alpha * (self.v_y + self.a_y*dt)
        self.t_last = t_start

    @property
    def v_xy(self):
        return self.v_x, - self.v_y

    @property
    def a_xy(self):
        return self.a_x, - self.a_y
