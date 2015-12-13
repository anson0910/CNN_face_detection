"""
Attempt to perform soft quantizing on face 12 net (with params -q, 0, q)
Must be executed under caffe-master folder
Training results will be saved as described in solver.prototxt
"""

import numpy as np
import sys
sys.path.append('/home/anson/PycharmProjects/face_net_surgery')
from quantize_functions import *

netKind = 24
quantizeBitNum = 5
stochasticRounding = False

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load trained high precision params ======================================
MODEL_FILE = '/home/anson/caffe-master/models/face_' + str(netKind) + '_cal/train_val.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_' + str(netKind) \
             + '_cal/face_' + str(netKind) + '_cal_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)
# ============ should be modified for different files ================
if netKind == 12:
    params = ['conv1', 'fc2', 'fc3']
elif netKind == 24:
    params = ['conv1', 'fc2', 'fc3']
elif netKind == 48:
    params = ['conv1', 'conv2', 'fc3', 'fc4']
# =====================================================================
# fc_params = {name: (weights, biases)}
original_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

# ==================  load file to save quantized parameters  =======================
MODEL_FILE = '/home/anson/caffe-master/models/face_' + str(netKind) + '_cal/train_val.prototxt'
if stochasticRounding:
    PRETRAINED = '/home/anson/caffe-master/models/face_' + str(netKind) + '_cal/face_' \
                 + str(netKind) + '_cal_soft_SRquantize_' + str(quantizeBitNum) +'.caffemodel'
else:
    PRETRAINED = '/home/anson/caffe-master/models/face_' + str(netKind) + '_cal/face_' \
                 + str(netKind) + '_cal_soft_quantize_' + str(quantizeBitNum) +'.caffemodel'
quantized_model = open(PRETRAINED, 'w')
net_quantized = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TRAIN)
params_quantized = params
# conv_params = {name: (weights, biases)}
quantized_params = {pr: (net_quantized.params[pr][0].data, net_quantized.params[pr][1].data) for pr in params_quantized}

# transplant params
for pr, pr_quantized in zip(params, params_quantized):
    quantized_params[pr_quantized][0].flat = original_params[pr][0].flat  # flat unrolls the arrays
    quantized_params[pr_quantized][1][...] = original_params[pr][1]

# =============== training =============================
# Training net is called net_training
# At each iteration:
#     1. Assign high precision params to net_training, and quantize
#     2. Perform 1 step in net_training
#     3. Update params diff to high precision params in net_quantized
# At last iteration, quantize high precision params
# ======================================================

solver = caffe.SGDSolver('/home/anson/caffe-master/models/face_' + str(netKind) + '_cal/solver.prototxt')
net_training = solver.net
params_train = params

# training_params[pr_train][0].flat = original_params[pr][0].flat  # flat unrolls the arrays
# training_params[pr_train][1][...] = original_params[pr][1]
for numOfIterations in range(400000):
    # if numOfIterations % 250 == 0:
    #     print "Current iteration : " + str(numOfIterations)
    # =========== 1. Assign high precision params to net_training, and quantize ===========
    quantized_params = {pr: (net_quantized.params[pr][0].data, net_quantized.params[pr][1].data) for pr in params_quantized}
    training_params = {pr: (net_training.params[pr][0].data, net_training.params[pr][1].data) for pr in params_train}
    # transplant params
    for pr_quantized, pr_train in zip(params_quantized, params_train):
        training_params[pr_train][0].flat = quantized_params[pr_quantized][0].flat  # flat unrolls the arrays
        training_params[pr_train][1][...] = quantized_params[pr_quantized][1]

    for k, v in net_training.params.items():
        filters_weights = net_training.params[k][0].data
        filters_bias = net_training.params[k][1].data

        # ============ should be modified for different files ================
        if netKind == 12:
            if k == 'conv1':
                a_weight = 0
                a_bias = -4
            elif k == 'fc2':
                a_weight = -2
                a_bias = 1
            elif k == 'fc3':
                a_weight = -5
                a_bias = 0

        elif netKind == 24:
            if k == 'conv1':
                a_weight = -1
                a_bias = -3
            elif k == 'fc2':
                a_weight = -3
                a_bias = 0
            elif k == 'fc3':
                a_weight = -5
                a_bias = 0

        elif netKind == 48:
            if k == 'conv1':
                a_weight = -1
                a_bias = -6
            elif k == 'conv2':
                a_weight = -1
                a_bias = -6
            elif k == 'fc3':
                a_weight = -4
                a_bias = 1
            elif k == 'fc4':
                a_weight = -6
                a_bias = 0

        # =====================================================================
        b_weight = quantizeBitNum - 1 - a_weight
        b_bias = quantizeBitNum - 1 - a_bias
        # lists of all possible values under current quantized bit num
        if quantizeBitNum == 2:
            weightFixedPointList = tri_section_points(a_weight)
            biasFixedPointList = tri_section_points(a_bias)
        else:
            weightFixedPointList = fixed_point_list(a_weight, b_weight)
            biasFixedPointList = fixed_point_list(a_bias, b_bias)

        for currentNum in np.nditer(filters_weights, op_flags=['readwrite']):
            currentNum[...] = round_number(currentNum[...], weightFixedPointList, stochasticRounding)

        for currentNum in np.nditer(filters_bias, op_flags=['readwrite']):
            currentNum[...] = round_number(currentNum[...], biasFixedPointList, stochasticRounding)
    # ============ End of rounding params ================
    # =========== 2. Perform 1 step in net_training ===========
    solver.step(1)
    # =========== 3. Update params diff to high precision params in net_quantized ===========
    for k, v in net_training.params.items():
        # print (k, v[0].data.shape)
        net_quantized.params[k][0].data[...] -= net_training.params[k][0].diff
        net_quantized.params[k][1].data[...] -= net_training.params[k][1].diff
        filters_weights = net_quantized.params[k][0].data
        filters_bias = net_quantized.params[k][1].data

    # if numOfIterations % 10000 == 0:
    #     net_quantized.save('/home/anson/caffe-master/models/face_12c/face_12c_soft_quantize_' \
    #              + str(quantizeBitNum) + '_' + str(numOfIterations) + '.caffemodel')



