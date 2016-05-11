"""
Evaluation of quantized weight version of CNN cascade on FDDB dataset
"""

import numpy as np
import cv2
import time
import sys
from operator import itemgetter
from load_model_functions import *
from face_detection_functions import *

stochasticRoundedParams = False

quantizeBitNum = 0

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load models  ======================================
net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal = \
    load_face_models(quantizeBitNum, stochasticRoundedParams, loadNet=True, softQuantize=True)
# =======================================================================

total_time = 0
total_images = 0

# load and open files to read and write
for current_file in range(1, 11):

    print 'Processing file ' + str(current_file) + ' ...'

    read_file_name = './FDDB-fold/FDDB-fold-' + str(current_file).zfill(2) + '.txt'
    write_file_name = './detections/fold-' + str(current_file).zfill(2) + '-out.txt'
    write_file = open(write_file_name, "w")

    with open(read_file_name, "r") as ins:
        array = []
        for line in ins:
            array.append(line)      # list of strings

    number_of_images = len(array)

    for current_image in range(number_of_images):
        if current_image % 10 == 0:
            print 'Processing image : ' + str(current_image)
        # load image and convert to gray
        read_img_name = '/home/anson/FDDB/originalPics/' + array[current_image].rstrip() + '.jpg'
        img = cv2.imread(read_img_name)     # BGR

        min_face_size = 40
        stride = 5

        start = time.clock()

        img_forward = np.array(img, dtype=np.float32)
        img_forward -= np.array((104.00698793, 116.66876762, 122.67891434))

        rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size, stride, True)     # detect faces
        # print("Number of faces after 12 net: " + str(len(rectangles)))
        rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_24c_net(net_24c, img_forward, rectangles)
        # print("Number of faces after 24 net: " + str(len(rectangles)))
        rectangles = cal_face_24c_net(net_24_cal, img_forward, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_48c_net(net_48c, img_forward, rectangles)
        # print("Number of faces after 48 net: " + str(len(rectangles)))
        rectangles = globalNMS(rectangles)      # apply global NMS
        # print("Number of faces after NMS: " + str(len(rectangles)))
        rectangles = cal_face_48c_net(net_48_cal, img_forward, rectangles)      # calibration
        # print("Number of faces after 48 cal: " + str(len(rectangles)))

        end = time.clock()
        total_time += (end - start)
        total_images += 1

        number_of_faces = len(rectangles)
        # for rectangle in rectangles:    # draw rectangles
        #     cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (255, 0, 0), 2)

        # write to file
        write_file.write(array[current_image])
        write_file.write("{}\n".format( str(number_of_faces) ) )
        for i in range(number_of_faces):
            write_file.write( "{} {} {} {} {}\n".format(str(rectangles[i][0]), str(rectangles[i][1]),
                                                        str(rectangles[i][2] - rectangles[i][0]),
                                                        str(rectangles[i][3] - rectangles[i][1]),
                                                        str(rectangles[i][4])))

    write_file.close()
    print 'Average time spent on one image : ' + str(total_time / total_images) + ' s'
