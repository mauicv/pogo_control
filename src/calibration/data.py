from dataclasses import dataclass, field


@dataclass
class SensorDataArray:
    xs: list[int] = field(default_factory=list)
    acc_xs: list[float] = field(default_factory=list)
    acc_ys: list[float] = field(default_factory=list)
    acc_zs: list[float] = field(default_factory=list)
    gyro_xs: list[float] = field(default_factory=list)
    gyro_ys: list[float] = field(default_factory=list)
    gyro_zs: list[float] = field(default_factory=list)
    x_ints: list[float] = field(default_factory=list)
    y_ints: list[float] = field(default_factory=list)
    z_ints: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.xs.append(0)
        self.acc_xs.append(0)
        self.acc_ys.append(0)
        self.acc_zs.append(0)
        self.gyro_xs.append(0)
        self.gyro_ys.append(0)
        self.gyro_zs.append(0)
        self.x_ints.append(0)
        self.y_ints.append(0)
        self.z_ints.append(0)

    def update(self, data: list[float]):
        acc_x, acc_y, acc_z = data[0:3]
        gyro_x, gyro_y, gyro_z = data[3:6]
        x_int, y_int, z_int = self.x_ints[-1] + acc_x, self.y_ints[-1] + acc_y, self.z_ints[-1] + acc_z
        self.xs.append(len(self.xs))
        self.acc_xs.append(acc_x)
        self.acc_ys.append(acc_y)
        self.acc_zs.append(acc_z)
        self.gyro_xs.append(gyro_x)
        self.gyro_ys.append(gyro_y)
        self.gyro_zs.append(gyro_z)
        self.x_ints.append(x_int)
        self.y_ints.append(y_int)
        self.z_ints.append(z_int)

    def get_data(self, limit=100):
        return (
            self.xs[-limit:],
            self.acc_xs[-limit:],
            self.acc_ys[-limit:],
            self.acc_zs[-limit:],
            self.gyro_xs[-limit:],
            self.gyro_ys[-limit:],
            self.gyro_zs[-limit:],
            self.x_ints[-limit:],
            self.y_ints[-limit:],
            self.z_ints[-limit:]
        )

