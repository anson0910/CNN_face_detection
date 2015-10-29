import os

data_base_dir = "/home/anson/face_pictures/ILSVRC2014_train_0000"    # file containing pictures
read_file_dir = '/home/anson/face_pictures/suspicious'

image_file_list = []    # list of image names to delete
for file in os.listdir(read_file_dir):
    if file.endswith(".JPEG"):
        image_file_list.append(file)

number_of_images = len(image_file_list)

print number_of_images

current_image = 0
for current_image in range(number_of_images):
    if current_image % 10 == 0:
        print 'Processing image : ' + str(current_image)
    # print file_number
    image_dir = data_base_dir + '/' + image_file_list[current_image]
    # print image_file_list[current_image]
    # print image_dir
    if not os.path.isfile(image_dir):
        continue
    os.remove(image_dir)
