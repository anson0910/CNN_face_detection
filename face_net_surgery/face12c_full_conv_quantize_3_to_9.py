import numpy as np
import sys
from quantize_functions import *

stochasticRounding = True

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load face12c_full_conv  ======================================
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
params = ['conv1', 'fc2-conv', 'fc3-conv']
# fc_params = {name: (weights, biases)}
original_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for quantize_bit_num in range(3, 10):
    # ==================  load file to save quantized parameters  =======================
    MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
    if stochasticRounding:
        PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv_SRquantize_' \
                     + str(quantize_bit_num) +'.caffemodel'
    else:
        PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv_quantize_' \
                     + str(quantize_bit_num) +'.caffemodel'
    quantized_model = open(PRETRAINED, 'w')
    net_quantized = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    params_quantized = params
    # conv_params = {name: (weights, biases)}
    quantized_params = {pr: (net_quantized.params[pr][0].data, net_quantized.params[pr][1].data) for pr in params_quantized}

    print "\n============face12c_full_conv================="

    # transplant
    for pr, pr_quantized in zip(params, params_quantized):
        quantized_params[pr_quantized][0].flat = original_params[pr][0].flat  # flat unrolls the arrays
        quantized_params[pr_quantized][1][...] = original_params[pr][1]

    for k, v in net_quantized.params.items():
        print (k, v[0].data.shape)
        filters_weights = net_quantized.params[k][0].data
        filters_bias = net_quantized.params[k][1].data

        if k == 'conv1':
            a_weight = -1
            a_bias = -1
        elif k == 'fc2-conv':
            a_weight = -3
            a_bias = 2
        elif k == 'fc3-conv':
            a_weight = -4
            a_bias = 1

        b_weight = quantize_bit_num - 1 - a_weight
        b_bias = quantize_bit_num - 1 - a_bias

        # lists of all possible values under current quantized bit num
        weightFixedPointList = fixed_point_list(a_weight, b_weight)
        biasFixedPointList = fixed_point_list(a_bias, b_bias)

        # print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
        # print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
        # print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
        # print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))
        print "Quantizing to " + str(quantize_bit_num) + " bits."

        for currentNum in np.nditer(filters_weights, op_flags=['readwrite']):
            currentNum[...] = round_number(currentNum[...], weightFixedPointList, stochasticRounding)

        for currentNum in np.nditer(filters_bias, op_flags=['readwrite']):
            currentNum[...] = round_number(currentNum[...], biasFixedPointList, stochasticRounding)

    net_quantized.save(PRETRAINED)

    quantized_model.close()
