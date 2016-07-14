import os
import cv2
import random

# data_base_dir = "/home/anson/face_pictures/CACD2000"    # file containing pictures
data_base_dir = "/media/anson/082EA3B42EA39968/Documents and Settings/Anson/Desktop/face_datasets/CACD2000"
save_dir = '/home/anson/face_pictures/calibration/cal_'    # start of dir to save pictures

image_file_list = []
for file in os.listdir(data_base_dir):
    if file.endswith(".jpg"):
        image_file_list.append(file)

random.shuffle(image_file_list)     # shuffle list

number_of_pictures = len(image_file_list)     # images in current folder
print number_of_pictures

width = 250
height = 250

y = height * 0.25
x = width * 0.25
h = height * 0.5
w = width * 0.5

# =========== Create calibration files ==========
for current_dir in range(45):
    cur_dir_name = save_dir + str(current_dir).zfill(2)
    os.mkdir(cur_dir_name)

# =========== Start processing ===============

for current_image in range(40000):     # start to save from calXX_000000.jpg
    if current_image % 200 == 0:
        write_content = 'Processing image ' + str(current_image) + '\n'
        print write_content
    image_file_name = image_file_list[current_image]
    read_img_name = data_base_dir + '/' + image_file_name
    img = cv2.imread(read_img_name)

    current_label = 0       # start from first label (s = 0.83, x = -0.17, y = -0.17)

    for cur_scale in [0.83, 0.91, 1.0, 1.10, 1.21]:
        for cur_x in [-0.17, 0, 0.17]:
            for cur_y in [-0.17, 0, 0.17]:
                s_n = 1 / cur_scale
                x_n = -cur_x / cur_scale
                y_n = -cur_y / cur_scale

                x_temp = x - (x_n * w / s_n)
                y_temp = y - (y_n * h / s_n)
                w_temp = w / s_n
                h_temp = h / s_n

                cropped_img = img[y_temp:y_temp+h_temp, x_temp:x_temp+w_temp]

                save_image_name = save_dir + str(current_label).zfill(2) + '/cal' + str(current_label).zfill(2) + '_' + \
                                    str(current_image).zfill(6) + '.jpg'

                cv2.imwrite(save_image_name, cropped_img)   # save cropped image
                current_label += 1


