import os

for i, file in enumerate(os.listdir("calibrate")):
    os.rename(f"./calibrate/{file}", f'./calibrate/calibrate-{i}.jpg')
