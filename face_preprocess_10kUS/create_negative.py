import numpy as np
import cv2
import os

data_base_dir = "/home/anson/face_pictures/ILSVRC2014_train_0000"     # file containing pictures
start_neg_dir = 1
end_neg_dir = 50
file_list = []      # list of strings storing names of pictures

for file in os.listdir(data_base_dir):
    if file.endswith(".JPEG"):
        file_list.append(file)

number_of_pictures = len(file_list)     # 5546 pictures

# ============== create directories ==================================
directory = '/home/anson/face_pictures/negatives/negative_'    # start of path

for cur_file in range(1, 50):

    path = directory + str(cur_file).zfill(2)

    if not os.path.exists(path):
        os.makedirs(path)
# ============== create negatives =====================================
for current_neg_dir in range(start_neg_dir, end_neg_dir + 1):
    save_image_number = 0
    save_dir_neg = "/home/anson/face_pictures/negatives/negative_" + str(current_neg_dir).zfill(2)    # file to save patches

    for current_image in range((current_neg_dir - 1)*300, (current_neg_dir - 1)*300 + 300):    # take 300 images
        if current_image % 100 == 0:
            print "Processing image number " + str(current_image)
        read_img_name = data_base_dir + '/' + file_list[current_image].strip()
        img = cv2.imread(read_img_name)     # read image
        height, width, channels = img.shape

        crop_size = min(height, width) / 2  # start from half of shorter side

        while crop_size >= 12:
            for start_height in range(0, height, 100):
                for start_width in range(0, width, 100):
                    if (start_width + crop_size) > width:
                        break
                    cropped_img = img[start_height : start_height + crop_size, start_width : start_width + crop_size]
                    file_name = save_dir_neg + "/neg" + str(current_neg_dir).zfill(2) + "_" + str(save_image_number).zfill(6) + ".jpg"
                    cv2.imwrite(file_name, cropped_img)
                    save_image_number += 1
            crop_size *= 0.5

        if current_image == (number_of_pictures - 1):
            break



