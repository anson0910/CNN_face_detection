import numpy as np
import sys

def load_face_models(quantizeBitNum = 0, stochasticRoundedParams = False, loadNet = False, softQuantize=False):
    '''
    Loads face detection models
    :param quantizeBitNum: number of bits to quantize, non quantized params are loaded when quantizeBitNum is 0
    :param stochasticRoundedParams: Stochastic rounded params are loaded when this is true
    :param loadNet: if true, all nets will be loaded as caffe.Net instead of caffe.Classifier
    :return: all 6 models as a tuple
    '''
    # ==================  caffe  ======================================
    caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    # ==================  load face12c_full_conv  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
    if softQuantize:
        PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv_soft_quantize_2.caffemodel'
    elif quantizeBitNum == 0:
        PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.caffemodel'
    else:
        if stochasticRoundedParams:
            PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv_SRquantize_' \
                     + str(quantizeBitNum) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv_quantize_' \
                         + str(quantizeBitNum) + '.caffemodel'
    caffe.set_mode_gpu()
    net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    # ==================  load face_12_cal  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/anson/caffe-master/models/face_12_cal/deploy.prototxt'
    if quantizeBitNum == 0:
        PRETRAINED = '/home/anson/caffe-master/models/face_12_cal/face_12_cal_train_iter_400000.caffemodel'
    else:
        if stochasticRoundedParams:
            PRETRAINED = '/home/anson/caffe-master/models/face_12_cal/face_12_cal_SRquantize_' \
                     + str(quantizeBitNum) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/face_12_cal/face_12_cal_quantize_' \
                         + str(quantizeBitNum) + '.caffemodel'
    caffe.set_mode_gpu()
    if loadNet:
        net_12_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    else:
        net_12_cal = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(15, 15))
    # .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>
    # ==================  load face_24c  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/anson/caffe-master/models/face_24c/deploy.prototxt'
    if softQuantize:
        PRETRAINED = '/home/anson/caffe-master/models/face_24c/face_24c_soft_quantize_2.caffemodel'
    elif quantizeBitNum == 0:
        PRETRAINED = '/home/anson/caffe-master/models/face_24c/face_24c_train_iter_400000.caffemodel'
    else:
        if stochasticRoundedParams:
            PRETRAINED = '/home/anson/caffe-master/models/face_24c/face_24c_SRquantize_' \
                     + str(quantizeBitNum) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/face_24c/face_24c_quantize_' \
                         + str(quantizeBitNum) + '.caffemodel'
    caffe.set_mode_gpu()
    if loadNet:
        net_24c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    else:
        net_24c = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(30, 30))
    # .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>
    # ==================  load face_24_cal  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/anson/caffe-master/models/face_24_cal/deploy.prototxt'
    if quantizeBitNum == 0:
        PRETRAINED = '/home/anson/caffe-master/models/face_24_cal/face_24_cal_train_iter_400000.caffemodel'
    else:
        if stochasticRoundedParams:
            PRETRAINED = '/home/anson/caffe-master/models/face_24_cal/face_24_cal_SRquantize_' \
                     + str(quantizeBitNum) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/face_24_cal/face_24_cal_quantize_' \
                         + str(quantizeBitNum) + '.caffemodel'
    caffe.set_mode_gpu()
    if loadNet:
        net_24_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    else:
        net_24_cal = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(30, 30))
    # .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>
    # ==================  load face_48c  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/anson/caffe-master/models/face_48c/deploy.prototxt'
    if softQuantize:
        PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_soft_quantize_2.caffemodel'
    elif quantizeBitNum == 0:
        PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_train_iter_200000.caffemodel'
    else:
        if stochasticRoundedParams:
            PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_SRquantize_' \
                     + str(quantizeBitNum) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_quantize_' \
                         + str(quantizeBitNum) + '.caffemodel'
    caffe.set_mode_gpu()
    if loadNet:
        net_48c = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    else:
        net_48c = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(60, 60))
    # .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>
    # ==================  load face_48_cal  ======================================
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = '/home/anson/caffe-master/models/face_48_cal/deploy.prototxt'
    if softQuantize:
        PRETRAINED = '/home/anson/caffe-master/models/face_48_cal/face_48_cal_soft_quantize_2.caffemodel'
    elif quantizeBitNum == 0:
        PRETRAINED = '/home/anson/caffe-master/models/face_48_cal/face_48_cal_train_iter_300000.caffemodel'
    else:
        if stochasticRoundedParams:
            PRETRAINED = '/home/anson/caffe-master/models/face_48_cal/face_48_cal_SRquantize_' \
                     + str(quantizeBitNum) + '.caffemodel'
        else:
            PRETRAINED = '/home/anson/caffe-master/models/face_48_cal/face_48_cal_quantize_' \
                         + str(quantizeBitNum) + '.caffemodel'
    caffe.set_mode_gpu()
    if loadNet:
        net_48_cal = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
    else:
        net_48_cal = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(60, 60))
    # .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>

    return net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal

