from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpeedDataArray:
    speeds: list[float] = field(default_factory=list)
    ts: list[float] = field(default_factory=list)
    detection_flags: list[int] = field(default_factory=list)
    pogo_a: list[float] = field(default_factory=list)
    pogo_v: list[float] = field(default_factory=list)
    alpha: float = field(default=0.8)
    
    def update(self, speed: float, ts: float, pogo_a: float):
        self.speeds.append(speed)
        if len(self.ts) > 1 and self.ts[-1] == ts:
            self.detection_flags.append(0)
        else:
            self.detection_flags.append(1)
        self.ts.append(ts)
        self.pogo_a.append(pogo_a)
        v = (self.pogo_v[-1] + pogo_a) * self.alpha + speed * (1 - self.alpha)
        self.pogo_v.append(v)

    def get_data(self, limit=100):
        return (
            self.speeds[-limit:],
            self.detection_flags[-limit:],
            self.pogo_a[-limit:],
            self.pogo_v[-limit:],
        )
    

@dataclass
class SensorDataArray:
    acc_xs: list[float] = field(default_factory=list)
    acc_ys: list[float] = field(default_factory=list)
    acc_zs: list[float] = field(default_factory=list)
    gyro_xs: list[float] = field(default_factory=list)
    gyro_ys: list[float] = field(default_factory=list)
    gyro_zs: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.acc_xs.append(0)
        self.acc_ys.append(0)
        self.acc_zs.append(0)
        self.gyro_xs.append(0)
        self.gyro_ys.append(0)
        self.gyro_zs.append(0)

    def update(self, data: list[float]):
        acc_x, acc_y, acc_z = data[0:3]
        gyro_x, gyro_y, gyro_z = data[3:6]
        self.acc_xs.append(acc_x)
        self.acc_ys.append(acc_y)
        self.acc_zs.append(acc_z)
        self.gyro_xs.append(gyro_x)
        self.gyro_ys.append(gyro_y)
        self.gyro_zs.append(gyro_z)

    def get_data(self, limit=100):
        return (
            self.acc_xs[-limit:],
            self.acc_ys[-limit:],
            self.acc_zs[-limit:],
            self.gyro_xs[-limit:],
            self.gyro_ys[-limit:],
            self.gyro_zs[-limit:],
        )


@dataclass
class StateDataArray:
    pitch: list[float] = field(default_factory=list)
    roll: list[float] = field(default_factory=list)
    overturned: list[float] = field(default_factory=list)

    def update(
            self,
            overturned,
            pitch,
            roll,
        ):
        self.overturned.append(overturned)
        self.pitch.append(pitch)
        self.roll.append(roll)

    def get_data(self, limit=100):
        return (
            self.overturned[-limit:],
            self.pitch[-limit:],
            self.roll[-limit:],
        )

