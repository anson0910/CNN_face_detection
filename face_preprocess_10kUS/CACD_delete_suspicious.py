import os

data_base_dir = '/home/anson/face_pictures/CACD2000'
read_file_dir = '/home/anson/face_pictures/suspicious_CACD'

image_file_list = []    # list of image names to delete
for file in os.listdir(read_file_dir):
    if file.endswith(".jpg"):
        image_file_list.append(file)

number_of_images = len(image_file_list)

print number_of_images

current_image = 0
for current_image in range(number_of_images):
    file_number = image_file_list[current_image][3:5]   # file number (1~15)
    # print file_number
    image_dir = data_base_dir + '/' + image_file_list[current_image]
    # print image_file_list[current_image]
    # print image_dir
    if not os.path.isfile(image_dir):
        continue
    os.remove(image_dir)
    print 'Image ' + image_file_list[current_image] + ' deleted.'
