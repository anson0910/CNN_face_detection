"""
Transfers soft quantized params to net, to see performance on validation set
"""

import numpy as np
import sys

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load soft quantized params  ======================================
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_soft_quantize_2.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params = ['conv1', 'fc2', 'fc3']
# fc_params = {name: (weights, biases)}
original_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}


# ==================  load file to save quantized parameters  =======================
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_train_iter_10000.caffemodel'

quantized_model = open(PRETRAINED, 'w')
net_quantized = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params_quantized = params
# conv_params = {name: (weights, biases)}
quantized_params = {pr: (net_quantized.params[pr][0].data, net_quantized.params[pr][1].data) for pr in params_quantized}

# ================== transplant params =============================================
for pr, pr_quantized in zip(params, params_quantized):
    quantized_params[pr_quantized][0].flat = original_params[pr][0].flat  # flat unrolls the arrays
    quantized_params[pr_quantized][1][...] = original_params[pr][1]

net_quantized.save(PRETRAINED)


# ============== See paramter ranges =======================
for k, v in net_quantized.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net_quantized.params[k][0].data
    filters_bias = net_quantized.params[k][1].data
    #
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))
