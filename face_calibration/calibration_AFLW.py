import cv2
import os

data_base_dir = "/home/anson/face_pictures/AFLW/aflw_images"     # file containing pictures
read_file_name_faces = "/home/anson/face_pictures/AFLW/AFLW_Faces.txt"   # file saving face ID and corresponding files
read_file_name_rect = "/home/anson/face_pictures/AFLW/AFLW_Rect.txt"    # file saving rect info corresponding to face ID

faceID_fileName_dict = {}   # dictionary of str to str

# =========== read face file ===============
with open(read_file_name_faces, "r") as ins:
    array_faces = []
    for line in ins:
        line = line.replace(',', ' ')   # get rid of commas, and quotes
        array_faces.append(line.replace('"', ''))      # list of strings
array_faces = array_faces[1:]   # ignore header
number_of_lines = len(array_faces)

# =========== construct dictionary ( face ID to image name )===============
for current_line in range(number_of_lines):
    face_ID = array_faces[current_line].strip()[0:5]
    image_file_name = array_faces[current_line].strip()[6:20]
    faceID_fileName_dict[face_ID] = image_file_name

# =========== read rect file ===============
with open(read_file_name_rect, "r") as ins:
    array_rect = []
    for line in ins:
        line = line.replace(',', ' ')   # get rid of commas, and quotes
        array_rect.append(line.replace('"', ''))      # list of strings
array_rect = array_rect[1:]   # ignore header
number_of_lines = len(array_rect)

# =========== Create calibration files ==========
save_dir = '/home/anson/face_pictures/calibration/cal_'    # start of dir to save pictures

for current_dir in range(45):
    cur_dir_name = save_dir + str(current_dir).zfill(2)
    os.mkdir(cur_dir_name)

# =========== Start processing ===============

save_image_number = 0

for current_rect in range(0, number_of_lines):
    if current_rect % 10 == 0:
        print "Processing rect number " + str(current_rect)
    current_info = array_rect[current_rect].split()    # list of strings in the order of face_id x y w h ignore
    ID = current_info[0]
    x = max(0, int(current_info[1]))
    y = max(0, int(current_info[2]))
    w = int(current_info[3])
    h = int(current_info[4])

    # if (x == 0) or (y == 0):
    #     continue

    current_image_name = faceID_fileName_dict.get(ID)   # find corresponding image name
    # print current_image_name
    if current_image_name is None:
        continue
    read_img_name = data_base_dir + '/' + current_image_name
    if not os.path.exists(read_img_name):     # check if file exists
        continue
    img = cv2.imread(read_img_name)     # read image

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
                                            str(save_image_number).zfill(6) + '.jpg'
                current_label += 1

                if (x_temp < 0) or (y_temp < 0):
                    continue
                cv2.imwrite(save_image_name, cropped_img)   # save cropped image

    save_image_number += 1

print 'Created ' + str(save_image_number + 1) + ' calibrated images.'
