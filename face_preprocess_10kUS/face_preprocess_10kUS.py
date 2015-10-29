import numpy as np
import cv2
import os

data_base_dir = "/home/anson/face_pictures/Face Images"     # file containing pictures
save_dir = "/home/anson/face_pictures/positives/positive_01"         # file to save cropped faces
file_list = []      # list of strings storing names of pictures

for file in os.listdir(data_base_dir):
    if file.endswith(".jpg"):
        file_list.append(file)

number_of_pictures = len(file_list)     # 10168 pictures

# print number_of_pictures

for current_image in range(number_of_pictures):
    read_img_name = data_base_dir + '/' + file_list[current_image].strip()
    img = cv2.imread(read_img_name)     # read image
    height, width, channels = img.shape
    height_sixth = height/6
    width_eighth = width/8
    cropped_img = img[height_sixth : height - height_sixth, width_eighth : width - width_eighth]

    file_name = save_dir + "/pos01_" + str(current_image).zfill(6) + ".jpg"
    cv2.imwrite(file_name, cropped_img)
    # cv2.destroyAllWindows()