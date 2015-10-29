import numpy as np
import cv2
import os

data_base_dir = "/home/anson/face_pictures/positives"
start_dir = 4
end_dir = 4

for current_dir in range(start_dir, end_dir + 1):
    current_data_base_dir = data_base_dir + '/positive_' + str(current_dir).zfill(2)
    image_file_list = []
    for file in os.listdir(current_data_base_dir):
        if file.endswith(".jpg"):
            image_file_list.append(file)
    # print len(file_list)
    number_of_images = len(image_file_list)     # images in current folder
    print number_of_images

    for current_image in range(number_of_images):

        image_file_name = image_file_list[current_image]
        read_img_name = current_data_base_dir + '/' + image_file_name
        img = cv2.imread(read_img_name)     # read image

        if not os.path.isfile(read_img_name):
            print "Doesn't exist : " + image_file_name
            continue
        if img is None:
            print image_file_name + " is None."
            os.remove(read_img_name)
            continue
        height, width, channels = img.shape
        if height == 0 or width == 0:
            print image_file_name + " has 0 height/weight."
        elif height/float(width) > 3 or height/float(width) < 0.35:
            print image_file_name + " has extreme height/weight."
        #cv2.imshow('img',img)
