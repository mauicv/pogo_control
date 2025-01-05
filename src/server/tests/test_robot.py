from server.servo_controller import ServoController
from server.mpu6050Mixin import MPU6050Mixin
from server.servo import Servo
from time import sleep


def test_update_rates(robot):
    sleep(2)
    # NOTE: 2 servo updates per mpu update due to two servos, one interval recorded in the
    # mock gpio class will be close to the target interval the second will be very small hence
    # the mean will be close to the target interval.

    mean_servo_interval = sum(robot.gpio.intervals)/(len(robot.gpio.intervals)/2)
    assert mean_servo_interval - robot.servo_update_interval < 0.01
    mean_mpu_interval = sum(robot.mpu.intervals)/len(robot.mpu.intervals)
    assert mean_mpu_interval - robot.mpu_update_interval < 0.01


def test_servo_controller(robot):
    assert len(robot.get_data()) == 2 + 6
    sleep(2)
    assert abs(robot.servos[0]._value - (- 0.4)) < 0.01
    assert abs(robot.servos[1]._value - (- 0.4)) < 0.01
    assert abs(robot.get_servo_data()[0] - (- 0.4)) < 0.01
    assert abs(robot.get_servo_data()[1] - (- 0.4)) < 0.01
    robot.update_angle([0.1, 0.2])
    sleep(2)
    assert robot.servos[0]._value - 0.1 < 0.01
    assert robot.servos[1]._value - 0.2 < 0.01


def test_servo_limits(robot):
    robot.update_angle([0.4, 0.5])
    sleep(2)
    assert abs(robot.get_servo_data()[0] - 0.4) < 0.01
    assert abs(robot.get_servo_data()[1] - 0.5) < 0.01


