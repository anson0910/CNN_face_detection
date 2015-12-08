import sys
file_write = open('face_range.txt', 'w')
sys.stdout = file_write

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

# ==================  load face12c_full_conv  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print "\n============face12c_full_conv================="
#print [(k, v[0].data.shape) for k, v in net.params.items()]
for k, v in net.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net.params[k][0].data
    filters_bias = net.params[k][1].data
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))

# filters_conv1_weights = net.params['conv1'][0].data
# filters_conv1_bias = net.params['conv1'][1].data

# see caffe/examples/filter_visualization.ipynb

# ==================  load face_12_cal  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12_cal/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12_cal/face_12_cal_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print "\n============face_12_cal================="
#print [(k, v[0].data.shape) for k, v in net.params.items()]
for k, v in net.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net.params[k][0].data
    filters_bias = net.params[k][1].data
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))

# ==================  load face_24c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_24c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_24c/face_24c_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print "\n============face_24c================="
#print [(k, v[0].data.shape) for k, v in net.params.items()]
for k, v in net.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net.params[k][0].data
    filters_bias = net.params[k][1].data
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))

# ==================  load face_24_cal  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_24_cal/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_24_cal/face_24_cal_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print "\n============face_24_cal================="
#print [(k, v[0].data.shape) for k, v in net.params.items()]
for k, v in net.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net.params[k][0].data
    filters_bias = net.params[k][1].data
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))

# ==================  load face_48c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_48c/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_train_iter_200000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print "\n============face_48c================="
#print [(k, v[0].data.shape) for k, v in net.params.items()]
for k, v in net.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net.params[k][0].data
    filters_bias = net.params[k][1].data
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))

# ==================  load face_48_cal  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_48_cal/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_48_cal/face_48_cal_train_iter_390000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)

print "\n============face_48_cal================="
#print [(k, v[0].data.shape) for k, v in net.params.items()]
for k, v in net.params.items():
    # print (k, v[0].data.shape)
    filters_weights = net.params[k][0].data
    filters_bias = net.params[k][1].data
    print ("Shape of " + k + " weight params : " + str(filters_weights.shape))
    print ("Max : " + str(filters_weights.max()) + "  min : " + str(filters_weights.min()))
    print ("Shape of " + k + " bias params: " + str(filters_bias.shape))
    print ("Max : " + str(filters_bias.max()) + "  min : " + str(filters_bias.min()))

file_write.close()
