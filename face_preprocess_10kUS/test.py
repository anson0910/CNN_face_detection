import os

save_dir_neg = '/home/anson/face_pictures/negatives/negative_99'  # file to save patches

file_list = []      # list to save image names
for file in os.listdir(save_dir_neg):
    if file.endswith(".jpg"):
        file_list.append(file)

number_of_pictures = len(file_list)     # 9101 pictures

print number_of_pictures