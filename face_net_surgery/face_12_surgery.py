'''
Transplants parameters to full conv version of net
'''

import numpy as np
import cv2
import time
import os
from operator import itemgetter

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

softQuantize = True
quantizeBitNum = 2

# ==================  load face_12c  ======================================
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
if softQuantize:
    PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_soft_quantize_2.caffemodel'
else:
    PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

params = ['fc2', 'fc3']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)


# Load the fully convolutional network to transplant the parameters.
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
if softQuantize:
    PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_soft_quantize_2.caffemodel'
else:
    PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_train_iter_400000.caffemodel'
net_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params_full_conv = ['fc2-conv', 'fc3-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

# transplant
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

if softQuantize:
    net_full_conv.save('/home/anson/caffe-master/models/face_12c/face12c_full_conv_soft_quantize_2.caffemodel')
else:
    net_full_conv.save('/home/anson/caffe-master/models/face_12c/face12c_full_conv.caffemodel')
