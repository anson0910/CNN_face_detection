import os
import random

trainingNet = 24

data_base_dir = "/home/anson/face_pictures/negatives"     # directory containing files of positives

if trainingNet == 12:
    start_neg_dir = 1
    end_neg_dir = 33
    # load and open files to read and write
    write_file_name = '/home/anson/caffe-master/data/face/all_negatives.txt'
elif trainingNet == 24:
    start_neg_dir = 99
    end_neg_dir = 99
    # load and open files to read and write
    write_file_name = '/home/anson/caffe-master/data/face/all_negatives_24c.txt'
elif trainingNet == 48:
    start_neg_dir = 98
    end_neg_dir = 98
    # load and open files to read and write
    write_file_name = '/home/anson/caffe-master/data/face/all_negatives_48c.txt'


write_file = open(write_file_name, "w")

file_list = []      # list to save image names

for current_neg_dir in range(start_neg_dir, end_neg_dir + 1):
    current_dir = data_base_dir + '/negative_' + str(current_neg_dir).zfill(2)

    for file in os.listdir(current_dir):
        if file.endswith(".jpg"):
            write_name = current_dir + '/' + file + ' ' + str(0)
            file_list.append(write_name)

random.shuffle(file_list)   # shuffle list
number_of_lines = len(file_list)
print number_of_lines

# write to file
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()
