import os
import random

data_base_dir = "/home/anson/face_pictures/positives"     # directory containing files of positives
start_pos_dir = 1
end_pos_dir = 15        # 16 and 17 save AFLW males and females
file_list = []      # list to save image names

# load and open files to read and write
write_file_name = '/home/anson/caffe-master/data/face/all_positives.txt'
write_file = open(write_file_name, "w")

for current_pos_dir in range(start_pos_dir, end_pos_dir + 1):
    current_dir = data_base_dir + '/positive_' + str(current_pos_dir).zfill(2)

    for file in os.listdir(current_dir):
        if file.endswith(".jpg"):
            write_name = current_dir + '/' + file + ' ' + str(1)
            file_list.append(write_name)

random.shuffle(file_list)   # shuffle list
number_of_lines = len(file_list)

# write to file
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()



