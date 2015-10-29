import subprocess
import os

out_files_dir = "/home/anson/PycharmProjects/face_detection/detections"
save_DiscROC_dir = "/home/anson/PycharmProjects/face_detection/detections/"


for current_file in range(1, 11):
    # run evaluation
    subprocess.call(["/home/anson/FDDB/evaluation/evaluation/evaluate",
                           "-a", "/home/anson/FDDB/FDDB-folds/FDDB-fold-" + str(current_file).zfill(2) + "-ellipseList.txt",
                           "-d", out_files_dir + "/fold-" + str(current_file).zfill(2) + "-out.txt",
                           "-i", "/home/anson/FDDB/originalPics/",
                           "-l", "/home/anson/FDDB/FDDB-folds/FDDB-fold-" + str(current_file).zfill(2) + ".txt",
                           "-r", save_DiscROC_dir,
                           "-z", ".jpg"])

    # rename file
    os.rename("/home/anson/PycharmProjects/face_detection/detections/DiscROC.txt",
              "/home/anson/PycharmProjects/face_detection/detections/DiscROC-" + str(current_file).zfill(2) + ".txt")
