from src.server.camera import Picamera2Camera
import cv2

camera = Picamera2Camera(input_source="main")
array = camera.get_frame()
print(array.data.shape)
cv2.imwrite("test.png", array.data)
camera.close()
