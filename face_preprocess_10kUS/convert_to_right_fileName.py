import numpy as np
import cv2
import os

data_base_dir = "/home/anson/face_pictures/positives/positive_04"     # file containing pictures
save_dir = "/home/anson/face_pictures/positives/positive_05"         # file to save cropped faces

file_list = []
for file in os.listdir(data_base_dir):
    if file.endswith(".jpg"):
        file_list.append(file)
# print len(file_list)
number_of_pictures = len(file_list)     # 1682 images

for current_image in range(number_of_pictures):
    read_image_name = data_base_dir + '/' + file_list[current_image].strip()
    img = cv2.imread(read_image_name)
    save_image_name = save_dir + '/pos05_' + str(current_image).zfill(6) + '.jpg'
    cv2.imwrite(save_image_name, img)

