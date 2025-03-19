from picamera2 import Picamera2
from server.camera import Picamera2Camera
import cv2
import time
from tqdm import tqdm
import numpy as np
import glob
import json
from pprint import pprint


def take_calibration_images(interval: int = 1, number_of_images: int = 12):
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (2048, 1536)},
    )
    picam2.configure(config)
    picam2.start()

    for i in tqdm(range(number_of_images)):
        array = picam2.capture_array("main")
        cv2.imwrite(f"calibration-imgs/test_{i}.png", array)
        time.sleep(interval)

    picam2.close()


def calibrate_camera():
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    for fname in tqdm(glob.glob('calibration-imgs/*.png')):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7,7), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )

    data = {
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()
    }

    pprint(data)

    with open("calibration_matrix.json", "w") as f:
        json.dump(data, f)


def take_camera_image():
    with open("camera_calibration_files/picamera-module-3.json", "r") as f:
        c_params = json.load(f)
        camera_matrix = np.array(c_params['camera_matrix'])
        dist_coeff = np.array(c_params['dist_coeff'])

    camera = Picamera2Camera(
        input_source="main",
        height=1536,
        width=2048,
        camera_matrix=camera_matrix,
        dist_coeff=dist_coeff,
    )
    array = camera.get_frame()
    print(array.data.shape)
    cv2.imwrite("test.png", array.data)
    camera.close()