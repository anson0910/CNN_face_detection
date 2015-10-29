import cv2
import os

data_base_dir = "/home/anson/face_pictures/CACD2000"    # file containing pictures

image_file_list = []
for file in os.listdir(data_base_dir):
    if file.endswith(".jpg"):
        image_file_list.append(file)

number_of_pictures = len(image_file_list)     # images in current folder
print number_of_pictures

width = 250
height = 250

y = height * 0.25
x = width * 0.25
h = height * 0.5
w = width * 0.5

start_dir = 5
end_dir = 13
for cur_dir in range(start_dir, end_dir + 1):
    print 'Processing file ' + str(cur_dir)
    save_file_number = 0     # starting file to save in positives
    save_dir = "/home/anson/face_pictures/positives/positive_" + str(cur_dir).zfill(2)  # file to save cropped faces
    for current_image in range((cur_dir - 5) * 20000, (cur_dir - 5) * 20000 + 20000):
        if not os.path.exists(save_dir):    # create folder if doesn't exist
            os.makedirs(save_dir)
        if current_image % 2000 == 0:
            print 'Processing image ' + str(current_image)
        image_file_name = image_file_list[current_image]
        read_img_name = data_base_dir + '/' + image_file_name
        img = cv2.imread(read_img_name)

        cropped_img = img[y:y+h, x:x+w]

        file_name = save_dir + "/pos" + str(cur_dir).zfill(2) + "_" + str(save_file_number).zfill(6) + ".jpg"
        cv2.imwrite(file_name, cropped_img)

        save_file_number += 1

