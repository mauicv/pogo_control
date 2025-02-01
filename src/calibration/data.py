from dataclasses import dataclass, field


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
class PitchRollDataArray:
    pitch: list[float] = field(default_factory=list)
    roll: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.pitch.append(0)
        self.roll.append(0)

    def update(self, pitch, roll):
        self.pitch.append(pitch)
        self.roll.append(roll)

    def get_data(self, limit=100):
        return (
            self.pitch[-limit:],
            self.roll[-limit:],
        )


@dataclass
class VelocityDataArray:
    vx: list[float] = field(default_factory=list)
    vy: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.vx.append(0)
        self.vy.append(0)

    def update(self, vx, vy):
        self.vx.append(vx)
        self.vy.append(vy)

    def get_data(self, limit=100):
        return (
            self.vx[-limit:],
            self.vy[-limit:],
        )


@dataclass
class StateDataArray:
    v: list[float] = field(default_factory=list)
    r: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.v.append(0)
        self.r.append(0)

    def update(self, v, r):
        self.v.append(v)
        self.r.append(r)

    def get_data(self, limit=100):
        return (
            self.v[-limit:],
            self.r[-limit:],
        )
