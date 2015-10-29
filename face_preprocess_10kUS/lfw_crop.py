import numpy as np
import cv2
import os

save_file_number = 0   # starting file to save in positives
save_dir = "/home/anson/face_pictures/positives/positive_03"         # file to save cropped faces
data_base_dir = "/home/anson/face_pictures/lfw_crop"        # file containing pictures
image_file_list = []
for file in os.listdir(data_base_dir):
    if file.endswith(".ppm"):
        image_file_list.append(file)
# print len(file_list)
number_of_pictures = len(image_file_list)     # images in current folder

for current_image in range(number_of_pictures):
    print "Processing image number " + str(current_image)

    image_file_name = image_file_list[current_image]
    read_img_name = data_base_dir + '/' + image_file_name
    img = cv2.imread(read_img_name)     # read image

    # print ground_truth_info
    file_name = save_dir + "/pos03_" + str(save_file_number).zfill(6) + ".jpg"
    cv2.imwrite(file_name, img)
    # cv2.destroyAllWindows()
    # print read_img_name
    save_file_number += 1