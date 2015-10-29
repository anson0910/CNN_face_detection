import os
import random

data_base_dir = "/home/anson/face_pictures/calibration"     # directory containing files of positives
start_cal_dir = 0
end_cal_dir = 44
file_list = []      # list to save image names

# load and open files to read and write
write_file_name = '/home/anson/caffe-master/data/face/all_calibrations.txt'
write_file = open(write_file_name, "w")

for current_cal_dir in range(start_cal_dir, end_cal_dir + 1):
    current_dir = data_base_dir + '/cal_' + str(current_cal_dir).zfill(2)

    for file in os.listdir(current_dir):
        if file.endswith(".jpg"):
            write_name = current_dir + '/' + file + ' ' + str(current_cal_dir)  # file name + label
            file_list.append(write_name)

random.shuffle(file_list)   # shuffle list
number_of_lines = len(file_list)
print number_of_lines

# write to file
for current_line in range(number_of_lines):
    write_file.write(file_list[current_line] + '\n')

write_file.close()
