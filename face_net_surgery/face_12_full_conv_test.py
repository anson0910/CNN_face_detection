import numpy as np
from PIL import Image
import cv2
import time
import os
from operator import itemgetter

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# ==================  load face_12c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.caffemodel'
caffe.set_mode_gpu()
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# ==========================================================================
def find_initial_scale(net_kind, min_face_size):
    '''
    :param net_kind: what kind of net (12, 24, or 48)
    :param min_face_size: minimum face size
    :return:    returns scale factor
    '''
    return float(min_face_size) / net_kind
def resize_image(img, scale):
    '''
    :param img: original img
    :param scale: scale factor
    :return:    resized image
    '''
    height, width, channels = img.shape
    new_height = int(height / scale)     # resized new height
    new_width = int(width / scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim)      # resized image
    return img_resized
def detect_face_12c(caffe_img, min_face_size, stride, multiScale=False):
    '''
    :param img: image in caffe style to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :return:    list of rectangles after global NMS
    '''
    multiScale = True
    net_kind = 12
    scale_factor = 1.18
    rectangles = []   # list of rectangles [x11, y11, x12, y12, confidence, current_scale] (corresponding to original image)

    current_scale = find_initial_scale(net_kind, min_face_size)     # find initial scale
    caffe_img_resized = resize_image(caffe_img, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape

    while current_height > net_kind and current_width > net_kind:
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))  # switch from H x W x C to C x H x W
        # shape for input (data blob is N x C x H x W), set data
        net_12c_full_conv.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12c_full_conv.blobs['data'].data[...] = caffe_img_resized_CHW
        # run net and take argmax for prediction
        net_12c_full_conv.forward()
        # out = net_48c_full_conv.blobs['prob'].data[0].argmax(axis=0)
        out = net_12c_full_conv.blobs['prob'].data[0][1, :, :]
        out_height, out_width, channels = out.shape

        threshold = 0.03

        for current_y in range(0, out_height):
            for current_x in range(0, out_width):
                # total_windows += 1
                confidence = out[current_y, current_x]  # left index is y, right index is x (starting from 0)
                if confidence >= threshold:
                    current_rectangle = [int(2*current_x*current_scale), int(2*current_y*current_scale),
                                             int(2*current_x*current_scale + net_kind*current_scale),
                                             int(2*current_y*current_scale + net_kind*current_scale),
                                             confidence, current_scale]     # find corresponding patch on image
                    rectangles.append(current_rectangle)
        if multiScale is False:
            break
        else:
            caffe_img_resized = resize_image(caffe_img_resized, scale_factor)
            current_scale *= scale_factor
            current_height, current_width, channels = caffe_img_resized.shape

    return rectangles


image_file_name = '/home/anson/FDDB/originalPics/2002/07/19/big/img_391.jpg'

img = cv2.imread(image_file_name)     # BGR
img = np.array(img, dtype=np.float32)
img -= np.array((104.00698793, 116.66876762, 122.67891434))



start = time.clock()

min_face_size = 40
stride = 5

# print caffe_img_resized.shape


end = time.clock()
print 'Time spent : ' + str(end - start) + ' s'
#
# img = cv2.imread(image_file_name)
# print img.shape
# print out.shape
# np.set_printoptions(threshold='nan')
# print out
# cv2.imshow('img', out*255)
# cv2.waitKey(0)
