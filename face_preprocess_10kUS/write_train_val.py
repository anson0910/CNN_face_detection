import os
import cv2
import shutil
import random

trainingNet = 24

# load and open files to read and write
# =================== face_12c =================================
if trainingNet == 12:
    pos_file_name = '/home/anson/caffe-master/data/face/all_positives.txt'
    neg_file_name = '/home/anson/caffe-master/data/face/all_negatives.txt'
    train_file_name = '/home/anson/face_pictures/train_12c'
    val_file_name = '/home/anson/face_pictures/val'
    write_train_name = '/home/anson/caffe-master/data/face/train_12c.txt'
    write_train = open(write_train_name, "w")
    write_val_name = '/home/anson/caffe-master/data/face/val.txt'
    write_val = open(write_val_name, "w")

# =================== face_24c =================================
elif trainingNet == 24:
    pos_file_name = '/home/anson/caffe-master/data/face/all_positives.txt'
    neg_file_name = '/home/anson/caffe-master/data/face/all_negatives_24c.txt'
    train_file_name = '/home/anson/face_pictures/train_24c'
    val_file_name = '/home/anson/face_pictures/val'
    write_train_name = '/home/anson/caffe-master/data/face/train_24c.txt'
    write_train = open(write_train_name, "w")
    write_val_name = '/home/anson/caffe-master/data/face/val.txt'
    write_val = open(write_val_name, "w")

# =================== face_48c =================================
elif trainingNet == 48:
    pos_file_name = '/home/anson/caffe-master/data/face/all_positives.txt'
    neg_file_name = '/home/anson/caffe-master/data/face/all_negatives_48c.txt'
    train_file_name = '/home/anson/face_pictures/train_48c'
    val_file_name = '/home/anson/face_pictures/val'
    write_train_name = '/home/anson/caffe-master/data/face/train_48c.txt'
    write_train = open(write_train_name, "w")
    write_val_name = '/home/anson/caffe-master/data/face/val.txt'
    write_val = open(write_val_name, "w")


pos = []
with open(pos_file_name, "r") as ins:
    for line in ins:
        pos.append(line)      # list of positive file names and labels

neg = []
with open(neg_file_name, "r") as ins:
    for line in ins:
        neg.append(line)      # list of negative file names and labels

number_of_pos = len(pos)
number_of_neg = len(neg)

# take first 500 images of pos and neg as val
val = []
val[0:500] = pos[0:500]
val[500:1000] = neg[0:500]
random.shuffle(val)
for current_image in range(1000):
    source = val[current_image][0:-3]   # retrieve image file name (including directory) from val
    image_file_name = val[current_image][48:-3]     # retrieve image file name
    label = int(val[current_image][-2:-1])   # retrieve label
    destination = val_file_name
    shutil.copy(source, destination)    # copy image to val folder
    image_file_complete = destination + '/' + image_file_name
    # write_val.write(image_file_complete + ' ' + label + '\n')      # write to val.txt
    write_val.write(image_file_name + ' ' + str(label) + '\n')      # write to val.txt
write_val.close()

# train data
train = []
train[0:number_of_pos - 500] = pos[500:]  # all positives not in val are assigned to train
train[number_of_pos - 500:] = neg[500:]     # assign 450000 negatives to train
random.shuffle(train)
number_of_train_data = len(train)

# write to train.txt
for current_image in range(number_of_train_data):
    if current_image % 1000 == 0:
        print 'Processing training data : ' + str(current_image)
    # print train[current_image].strip()
    source = train[current_image][0:-3]   # retrieve image file name (including directory) from train
    image_file_name = train[current_image][48:-3]     # retrieve image file name
    label = train[current_image].strip()[-1:]   # retrieve label
    # print label
    destination = train_file_name
    shutil.copy2(source, destination)    # copy image to train folder
    # image_file_complete = destination + '/' + image_file_name
    # write_train.write(image_file_complete + ' ' + label + '\n')      # write to train.txt
    write_content = image_file_name + ' ' + label + '\n'
    # print write_content
    write_train.write(write_content)      # write to train.txt
write_train.close()














