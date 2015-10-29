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

# ==================  load face_48c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_48c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_train_iter_200000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

params = ['fc3', 'fc4']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)


# Load the fully convolutional network to transplant the parameters.
MODEL_FILE = '/home/anson/caffe-master/models/face_48c/face48c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_train_iter_200000.caffemodel'
net_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params_full_conv = ['fc3-conv', 'fc4-conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

# transplant
for pr, pr_conv in zip(params, params_full_conv):
    conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
    conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save('/home/anson/caffe-master/models/face_48c/face48c_full_conv.caffemodel')
