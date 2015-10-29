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

# ==================  load face_12c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

# =================== Load the quantized network to transplant the parameters.==============
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_quantize_8.caffemodel'
net_quantized = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print [(k, v[0].data.shape) for k, v in net.params.items()]

filters_conv1_weights = net.params['conv1'][0].data
filters_conv1_bias = net.params['conv1'][1].data

print filters_conv1_weights.shape
print filters_conv1_bias.shape

# print filters_conv1_weights
# print filters_conv1_bias

print filters_conv1_bias.max()
print filters_conv1_bias.min()

# see caffe/examples/filter_visualization.ipynb
