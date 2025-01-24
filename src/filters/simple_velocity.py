import math
import time
import numpy as np
GRAVITY = 9.80665
        
    
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
