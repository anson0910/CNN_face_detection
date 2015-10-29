import numpy as np
import cv2
import os

save_file_number = 7810   # starting file to save in positives
number_of_folders = 1208
save_dir = "/home/anson/face_pictures/positive_03"         # file to save cropped faces

for current_folder in range(740, number_of_folders + 1):
    data_base_dir = "/home/anson/face_pictures/colorferet/dvd2/data/images/" + str(current_folder).zfill(5)
    # file containing pictures
    print "Processing folder number " + str(current_folder)
    if not os.path.isdir(data_base_dir):    # check if directory exists
        continue

    image_file_list = []
    for file in os.listdir(data_base_dir):
        if file.endswith(".ppm"):
            image_file_list.append(file)
    # print len(file_list)
    number_of_pictures = len(image_file_list)     # images in current folder
    # print file_list[0]

    # current_image = 0
    for current_image in range(0, number_of_pictures):
        image_file_name = image_file_list[current_image]
        read_img_name = data_base_dir + '/' + image_file_name
        img = cv2.imread(read_img_name)     # read image

        # print ground_truth_info
        file_name = save_dir + "/" + str(save_file_number).zfill(6) + ".jpg"
        cv2.imwrite(file_name, img)
        # cv2.destroyAllWindows()
        # print read_img_name
        save_file_number += 1

