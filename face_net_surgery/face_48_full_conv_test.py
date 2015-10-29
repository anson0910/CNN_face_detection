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
# ==================  load face_48c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_48c/face48c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_48c/face48c_full_conv.caffemodel'
caffe.set_mode_gpu()
net_48c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

image_file_name = '/home/anson/FDDB/originalPics/2002/07/19/big/img_391.jpg'
im = Image.open(image_file_name)
img = cv2.imread(image_file_name)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
in_ = in_.transpose((2, 0, 1))

start = time.clock()

# shape for input (data blob is N x C x H x W), set data
net_48c_full_conv.blobs['data'].reshape(1, *in_.shape)
net_48c_full_conv.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net_48c_full_conv.forward()
# out = net_48c_full_conv.blobs['prob'].data[0].argmax(axis=0)
out = net_48c_full_conv.blobs['prob'].data[0][1, :, :]

threshold = 0.02
idx = out[:, :] >= threshold
out[idx] = 1
idx = out[:, :] < threshold
out[idx] = 0

end = time.clock()
print 'Time spent : ' + str(end - start)

print img.shape
print out.shape
print out

cv2.imshow('img', out*255)
cv2.waitKey(0)
