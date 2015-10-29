import os
import cv2
import shutil
import random
import re

# load and open files to read and write
read_file_name = '/home/anson/caffe-master/data/face/all_calibrations.txt'
train_file_name = '/home/anson/face_pictures/train_cal'
val_file_name = '/home/anson/face_pictures/val_cal'
write_train_name = '/home/anson/caffe-master/data/face/train_cal.txt'
write_train = open(write_train_name, "w")
write_val_name = '/home/anson/caffe-master/data/face/val_cal.txt'
write_val = open(write_val_name, "w")

cal = []
with open(read_file_name, "r") as ins:
    for line in ins:
        cal.append(line)      # list of positive file names and labels

number_of_cal = len(cal)

print number_of_cal

# take first 2000 images as validation set
val = []
val[0:2000] = cal[0:2000]
random.shuffle(val)
for current_image in range(2000):
    info = re.split('\s+', val[current_image].strip())
    source = info[0]   # retrieve image file name (including directory) from val
    image_file_name = info[0][-16:]     # retrieve image file name
    label = info[-1]  # retrieve label

    destination = val_file_name
    shutil.copy(source, destination)    # copy image to val folder
    os.remove(source)       # delete file
    # image_file_complete = destination + '/' + image_file_name
    # write_val.write(image_file_complete + ' ' + label + '\n')      # write to val.txt
    write_val.write(image_file_name + ' ' + str(label) + '\n')      # write to val.txt
write_val.close()

# train data
train = cal[2000:1000000]  # all positives not in val are assigned to train
random.shuffle(train)
number_of_train_data = len(train)

print 'Total training data : ' + str(number_of_train_data)
# write to train.txt
for current_image in range(number_of_train_data):
    if current_image % 100 == 0:
        print "Processing image number " + str(current_image)
    info = re.split('\s+', train[current_image].strip())
    source = info[0]   # retrieve image file name (including directory) from train
    image_file_name = info[0][-16:]     # retrieve image file name
    label = info[-1]  # retrieve label

    destination = train_file_name
    shutil.copy(source, destination)    # copy image to train folder
    os.remove(source)       # delete file
    # image_file_complete = destination + '/' + image_file_name
    # write_train.write(image_file_complete + ' ' + label + '\n')      # write to train.txt
    write_train.write(image_file_name + ' ' + str(label) + '\n')      # write to train.txt
write_train.close()













