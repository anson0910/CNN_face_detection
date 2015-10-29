import os

directory = '/home/anson/face_pictures/calibration/cal_'    # start of path

for cur_file in range(45):

    path = directory + str(cur_file).zfill(2)

    if not os.path.exists(path):
        os.makedirs(path)