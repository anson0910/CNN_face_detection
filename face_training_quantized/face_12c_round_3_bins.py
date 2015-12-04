"""
Rounds 3 bin weights in face_train_iter_xxx, and saves in face_soft_quantize_2
"""
import numpy as np
import sys
sys.path.append('/home/anson/PycharmProjects/face_net_surgery')
from quantize_functions import *

quantizeBitNum = 2
stochasticRounding = False

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load soft quantized params  ======================================
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_train_iter_150000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params = ['conv1', 'fc2', 'fc3']
# fc_params = {name: (weights, biases)}
original_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

# ==================  load file to save quantized parameters  =======================
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/deploy.prototxt'
if stochasticRounding:
    PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_soft_SRquantize_' \
                 + str(quantizeBitNum) +'.caffemodel'
else:
    PRETRAINED = '/home/anson/caffe-master/models/face_12c/face_12c_soft_quantize_' \
                 + str(quantizeBitNum) +'.caffemodel'
quantized_model = open(PRETRAINED, 'w')
net_quantized = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params_quantized = params
# conv_params = {name: (weights, biases)}
quantized_params = {pr: (net_quantized.params[pr][0].data, net_quantized.params[pr][1].data) for pr in params_quantized}

# transplant params
for pr, pr_quantized in zip(params, params_quantized):
    quantized_params[pr_quantized][0].flat = original_params[pr][0].flat  # flat unrolls the arrays
    quantized_params[pr_quantized][1][...] = original_params[pr][1]


# ================ Round params in net_quantized
for k, v in net_quantized.params.items():
    filters_weights = net_quantized.params[k][0].data
    filters_bias = net_quantized.params[k][1].data

    # ============ should be modified for different files ================
    if k == 'conv1':
        a_weight = -1
        a_bias = -3
    elif k == 'fc2':
        a_weight = -3
        a_bias = 0
    elif k == 'fc3':
        a_weight = -5
        a_bias = 0
    # =====================================================================
    # lists of all possible values under current quantized bit num
    weightTriPoints = tri_section_points(a_weight)
    biasTriPoints = tri_section_points(a_bias)

    for currentNum in np.nditer(filters_weights, op_flags=['readwrite']):
        currentNum[...] = round_number(currentNum[...], weightTriPoints, stochasticRounding)

    for currentNum in np.nditer(filters_bias, op_flags=['readwrite']):
        currentNum[...] = round_number(currentNum[...], biasTriPoints, stochasticRounding)


net_quantized.save(PRETRAINED)
quantized_model.close()