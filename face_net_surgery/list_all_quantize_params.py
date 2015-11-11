'''
List all params of face12c_full_conv_quantize_quantize_bit_num
'''

import numpy as np
import sys

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
# =================================================================================
modelDirList = ['face_12c', 'face_12_cal']
modelNameList = ['face12c_full_conv', 'face_12_cal']

for curModel in range(len(modelDirList)):
    for quantize_bit_num in range(3, 10):

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
        PRETRAINED = '/home/anson/caffe-master/models/' \
                     + modelDirList[curModel] + '/' + modelNameList[curModel] + '_quantize_' \
                     + str(quantize_bit_num) + '.caffemodel'
        caffe.set_mode_gpu()
        net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

        print "============" + modelNameList[curModel] + "_quantize_" + str(quantize_bit_num) + "================="
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
