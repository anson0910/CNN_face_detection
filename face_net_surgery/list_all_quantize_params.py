'''
List all params of face12c_full_conv_quantize_quantize_bit_num
'''

import numpy as np
import sys

stochasticRounding = True

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
# =================================================================================
modelDirList = ['face_12c', 'face_12_cal', 'face_24c', 'face_24_cal', 'face_48c', 'face_48_cal']
modelNameList = ['face12c_full_conv', 'face_12_cal', 'face_24c', 'face_24_cal', 'face_48c', 'face_48_cal']

for curModel in range(len(modelDirList)):
    for quantize_bit_num in range(3, 10):
        if stochasticRounding:
            file_write = open('./params/' + modelNameList[curModel] + '_SRquantize_'
                              + str(quantize_bit_num) + '_params.txt', 'w')
        else:
            file_write = open('./params/' + modelNameList[curModel] + '_quantize_'
                              + str(quantize_bit_num) + '_params.txt', 'w')
        sys.stdout = file_write
        # ==================  load face12c_full_conv  ======================================
        if curModel == 0:
            MODEL_FILE = '/home/anson/caffe-master/models/' \
                         + modelDirList[curModel] + '/' + modelNameList[curModel] + '.prototxt'
        else:
            MODEL_FILE = '/home/anson/caffe-master/models/' \
                         + modelDirList[curModel] + '/' + 'deploy.prototxt'
        if stochasticRounding:
            PRETRAINED = '/home/anson/caffe-master/models/' \
                     + modelDirList[curModel] + '/' + modelNameList[curModel] + '_SRquantize_' \
                     + str(quantize_bit_num) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/' \
                         + modelDirList[curModel] + '/' + modelNameList[curModel] + '_quantize_' \
                         + str(quantize_bit_num) + '.caffemodel'
        caffe.set_mode_gpu()
        net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
        if stochasticRounding:
            print "=========" + modelNameList[curModel] + "_SRquantize_" + str(quantize_bit_num) + "========"
        else:
            print "=========" + modelNameList[curModel] + "_quantize_" + str(quantize_bit_num) + "========"
        for k, v in net.params.items():
            print (k, v[0].data.shape)
            filters_weights = net.params[k][0].data
            filters_bias = net.params[k][1].data

            print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
            for currentNum in np.nditer(filters_weights):
                print currentNum

            print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
            for currentNum in np.nditer(filters_bias):
                print currentNum

        file_write.close()
