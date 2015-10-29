import numpy as np
import cv2
import os

data_base_dir = "/home/anson/face_pictures/AFLW/aflw_images"     # file containing pictures
save_dir_male = "/home/anson/face_pictures/positives/positive_16"         # file to save cropped male faces
save_dir_female = "/home/anson/face_pictures/positives/positive_17"         # file to save cropped female faces
read_file_name_faces = "/home/anson/face_pictures/AFLW/AFLW_Faces.txt"   # file saving face ID and corresponding files
read_file_name_rect = "/home/anson/face_pictures/AFLW/AFLW_Rect.txt"    # file saving rect info corresponding to face ID
read_file_name_sex = "/home/anson/face_pictures/AFLW/AFLW_sex.txt"    # file saving sex info corresponding to face ID

starting_image_number = 0
faceID_fileName_dict = {}   # dictionary of str to str
faceID_sex_dict = {}        # dictionary of str to str

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

# print faceID_fileName_dict.get('39341')
# print len(faceID_fileName_dict)

# =========== read sex file ===============
with open(read_file_name_sex, "r") as ins:
    array_sex = []
    for line in ins:
        line = line.replace(',', ' ')   # get rid of commas, and quotes
        array_sex.append(line.replace('"', ''))      # list of strings
array_sex = array_sex[1:]   # ignore header
number_of_lines = len(array_sex)

# =========== construct dictionary ( face ID to sex )===============
for current_line in range(number_of_lines):
    face_ID = array_sex[current_line].strip()[0:5]
    ID_sex = array_sex[current_line].strip()[6:7]
    faceID_sex_dict[face_ID] = ID_sex       # should be 'm' or 'f'

# =========== read rect file ===============
with open(read_file_name_rect, "r") as ins:
    array_rect = []
    for line in ins:
        line = line.replace(',', ' ')   # get rid of commas, and quotes
        array_rect.append(line.replace('"', ''))      # list of strings
array_rect = array_rect[1:]   # ignore header
number_of_lines = len(array_rect)

# =========== Start processing ===============
save_file_number_male = 0
save_file_number_female = 0

for current_rect in range(0, number_of_lines):
    if current_rect % 10 == 0:
        print "Processing rect number " + str(current_rect)
    current_info = array_rect[current_rect].split()    # list of strings in the order of face_id x y w h ignore
    ID = current_info[0]
    x = max(0, int(current_info[1]))
    y = max(0, int(current_info[2]))
    w = int(current_info[3])
    h = int(current_info[4])

    current_image_name = faceID_fileName_dict.get(ID)   # find corresponding image name
    current_image_sex = faceID_sex_dict.get(ID)   # find corresponding sex
    # print current_image_name
    if current_image_name is None:
        continue
    read_img_name = data_base_dir + '/' + current_image_name
    if not os.path.exists(read_img_name):     # check if file exists
        continue
    img = cv2.imread(read_img_name)     # read image
    cropped_img = img[y : y + h, x : x + w]

    if current_image_sex == 'm':        # check sex is male or female
        file_name = save_dir_male + "/pos16_" + str(save_file_number_male).zfill(6) + ".jpg"
        save_file_number_male += 1
    else:
        file_name = save_dir_female + "/pos17_" + str(save_file_number_female).zfill(6) + ".jpg"
        save_file_number_female += 1
    cv2.imwrite(file_name, cropped_img)
    # cv2.imshow('cropped img', cropped_img)
    # cv2.waitKey(0)