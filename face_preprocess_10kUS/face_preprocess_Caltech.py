import numpy as np
import cv2
import os

data_base_dir = "/home/anson/face_pictures/Caltech_7092faces/Caltech_WebFaces"     # file containing pictures
save_dir = "/home/anson/face_pictures/positives/positive_01"         # file to save cropped faces
read_file_name = "/home/anson/face_pictures/Caltech_7092faces/WebFaces_GroundTruth.txt"

starting_image_number = 10168

with open(read_file_name, "r") as ins:
    array = []
    for line in ins:
        array.append(line)      # list of strings

number_of_pictures = len(array)     # 10168 pictures
# print number_of_pictures

for current_image in range(number_of_pictures):
    save_file_number = starting_image_number + current_image

    image_file_name = array[current_image].split()[0]   # name of image file
    # print image_file_name
    ground_truth_info = [int(float(i)) for i in array[current_image].split()[1:]]
    # ignore first element of array[current_image].split(), because it contains name of image file
    # ground_truth_info contains Leye-x Leye-y Reye-x Reye-y nose-x nose-y mouth-x mouth-y
    eye_dist = abs(ground_truth_info[0] - ground_truth_info[2])     # eye x distance
    eye_mouth_dist =  abs(ground_truth_info[1] - ground_truth_info[7])      # eye and mouth y distance
    left_x = min(ground_truth_info[0], ground_truth_info[2], ground_truth_info[4], ground_truth_info[6])
    right_x = max(ground_truth_info[0], ground_truth_info[2], ground_truth_info[4], ground_truth_info[6])
    # find most left and right x values
    top_y = min(ground_truth_info[1], ground_truth_info[3], ground_truth_info[5], ground_truth_info[7])
    bot_y = max(ground_truth_info[1], ground_truth_info[3], ground_truth_info[5], ground_truth_info[7])
    # find most top and bottom y values
    crop_lt_x = left_x - eye_dist/2
    crop_lt_y = top_y - eye_mouth_dist/2
    crop_br_x = right_x + eye_dist/2
    crop_br_y = bot_y + eye_mouth_dist/2

    read_img_name = data_base_dir + '/' + image_file_name
    # print read_img_name
    img = cv2.imread(read_img_name)     # read image
    cropped_img = img[crop_lt_y : crop_br_y, crop_lt_x : crop_br_x]
    # print cropped_img.shape
    # print crop_lt_y
    # print crop_br_y
    # print crop_lt_x
    # print crop_br_x
    # cv2.imshow('img',img)
    # cv2.imshow('cropped img', cropped_img)
    # cv2.waitKey(0)
    # print ground_truth_info

    file_name = save_dir + "/pos01_" + str(save_file_number).zfill(6) + ".jpg"
    cv2.imwrite(file_name, cropped_img)
    # print file_name
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print read_img_name



