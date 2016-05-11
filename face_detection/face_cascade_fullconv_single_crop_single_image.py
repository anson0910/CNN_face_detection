import numpy as np
import cv2
import time
from operator import itemgetter
from load_model_functions import *
from face_detection_functions import *

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# ==================  load models  ======================================
net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal = \
    load_face_models(loadNet=True)

nets = (net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal)

# load image and convert to gray
read_img_name = '/home/anson/FDDB/originalPics/2002/07/19/big/img_352.jpg'
img = cv2.imread(read_img_name)     # BGR

print img.shape

min_face_size = 48
stride = 5

# caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
# caffe_image = caffe_image[:, :, (2, 1, 0)]
img_forward = np.array(img, dtype=np.float32)
img_forward -= np.array((104, 117, 123))

rectangles = detect_faces_net(nets, img_forward, min_face_size, stride, True, 2, 0.05)
for rectangle in rectangles:    # draw rectangles
        cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)


number_of_faces = len(rectangles)
