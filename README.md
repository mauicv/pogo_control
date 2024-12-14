# Pogo Control

This project has sampling logic and a control system for a [`puppy_pi`](https://www.hiwonder.com/products/puppypi?variant=40213129003095) robot affectionately named `Pogo` by my sister. The robot is controlled by a Raspberry Pi and a MPU6050 IMU sensor. There are three main components:

## `client.py`

This is a simple client that sends commands to the robot. It is responsible for parsing the commands and sending them to the robot via a socket connection.

## `server.py`

The server is run on the robot. It is responsible for reading and implementing the actions sent from the client and sending the sensor data to the client.

## `sampler.py`

The sampler performs a set of actions. Firstly Its responsible for performing the actor model rollouts via the client logic. During this process it stores the actions (the PWM values for each motor) and the observations (the readings from the IMU). It uploads the data to a google cloud bucket. It uses the latest actor model to generate the actions.

Secondly, it checks the google cloud bucket for new actor model versions. If there are new versions it will download them and use them to generate the actions during the next round of rollouts.

Note the actor model is trained externally and periodically uploaded to the google cloud bucket.


## Setting up Pogo

1. Set up OS on the Raspberry Pi.
2. Enable I2C interface using raspi-config.
3. pip install requirements/server.txt
4. Setup pigpio daemon