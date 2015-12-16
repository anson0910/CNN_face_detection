'''
creates hard negatives for face_24c
'''
import numpy as np
import cv2
import os
from operator import itemgetter
import sys
sys.path.append('/home/anson/PycharmProjects/face_detection')
from face_detection_functions import *

neg_database = '/home/anson/face_pictures/ILSVRC2014_train_0000'    # directory containing negative images
save_dir_neg = '/home/anson/face_pictures/negatives/negative_99'  # file to save patches

file_list = []      # list to save image names

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load face12c_full_conv  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv_soft_quantize_2.caffemodel'

caffe.set_mode_gpu()
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# ==================  caffe  ======================================
for file in os.listdir(neg_database):
    if file.endswith(".JPEG"):
        file_list.append(file)

number_of_pictures = len(file_list)     # 9101 pictures

# =========== evaluate faces ============

save_image_number = 0

for current_picture in range(0, number_of_pictures):
    if current_picture % 10 == 0:
        print 'Processing image : ' + str(current_picture)
    image_name = file_list[current_picture]
    image_file_name = neg_database + '/' + image_name

    img = cv2.imread(image_file_name)   # load image
    min_face_size = 40
    stride = 5

    if img is None:
        continue

    # caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
    # caffe_image = caffe_image[:, :, (2, 1, 0)]
    img_forward = np.array(img, dtype=np.float32)
    img_forward -= np.array((104, 117, 123))

    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size,
                                 stride, True, 1.414, 0.05)  # detect faces
    rectangles = localNMS(rectangles)      # apply local NMS

    for cur_rectangle in rectangles:
        cropped_img = img[cur_rectangle[1]:cur_rectangle[3], cur_rectangle[0]:cur_rectangle[2]]
        save_image_name = save_dir_neg + '/neg99_' + str(save_image_number).zfill(6) + '.jpg'
        cv2.imwrite(save_image_name, cropped_img)       # save cropped image
        save_image_number += 1
        if save_image_number > 999998:
            break
    if save_image_number > 999998:
        break

    print "Total images produced : " + str(save_image_number)


# import os
#
# save_dir_neg = '/home/anson/face_pictures/negatives/negative_99'  # file to save patches
#
# file_list = []      # list to save image names
# for file in os.listdir(save_dir_neg):
#     if file.endswith(".jpg"):
#         file_list.append(file)
#
# number_of_pictures = len(file_list)     # 9101 pictures
#
# print number_of_pictures

