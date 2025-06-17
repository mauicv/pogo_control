# Pogo Control

![](/assets/pogo-rollout.gif)

This project has sampling logic and a control system for a [`puppy_pi`](https://www.hiwonder.com/products/puppypi?variant=40213129003095) robot affectionately named `Pogo` by my sister. The robot is controlled by a Raspberry Pi and a MPU6050 IMU sensor.

__Note__: This repo was built mostly as a POC and as such the testing, maintance and documentation is minimal throughout. It's part of a larger project to train a Robot to walk using Model based RL (see [this blog post](https://mauicv.com/#/posts/real-world-model-rl) if your interested). Feel free to reach out if you have questions.

## Setting up Pogo

1. Set up OS on the Raspberry Pi.
2. Enable I2C interface using raspi-config.
3. pip install requirements/server.txt
4. Setup pigpio daemon: see [here](https://abyz.me.uk/rpi/pigpio/download.html)
5. install cv2 dependencies: `apt-get update && apt-get install ffmpeg libsm6 libxext6  -y`


## Utilization

Pogo uses [click](https://click.palletsprojects.com/en/stable/) for a CLI. Use:

```
pogo --help
```

to see all options.

__Starting Pogo__: On pogo's rpi:

```sh
pogo pogo start
```

On the camera rpi:

__Starting camera__: On 

```sh
pogo camera start
```

__Deploy__: Will deploy `plastic-thumb-nozzle-212` which is the current best solution trained. Runs for 500 steps.

```sh
pogo client --num-steps=500 deploy --name="plastic-thumb-nozzle-212"
```

__Sample__: Starts a sampling loop. Uses `strain-super-ring` as the google bucket to upload rollouts to and `plastic-thumb-nozzle` as the model bucket to check for new actors. No action noise or weight perturbation. Rolls out 200 steps each sample.

```sh
pogo client --name="strain-super-ring" --model-name="plastic-thumb-nozzle" --noise-range=0 0 --num-steps=200 --weight-range=0 0 sample
```

