"""
Evaluation of quantized signal version of CNN cascade on FDDB dataset
Quantize bit numbers are obtained from face_net_surgery/full_conv_blob_ranges.txt
"""

import numpy as np
import cv2
import time
import sys
from operator import itemgetter
sys.path.append('../face_net_surgery')
from load_model_functions import *
from face_detection_functions import *
from quantize_functions import *

stochasticRoundedParams = False
quantizeBitNum = 9

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
# ==================  load models  ======================================
net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal = \
    load_face_models(loadNet=True)

# ==========================================================================
def quantizeBlob(netToQuantize, blobToQuantize, a_blob, forwardBlob):
    '''
    Quantizes blobToQuantize in netToQuantize, and forwards net from layer forwardBlob
    '''
    blobToChange = netToQuantize.blobs[blobToQuantize].data
    b_blob = quantizeBitNum - 1 - a_blob
    # lists of all possible values under current quantized bit num
    blobFixedPointList = fixed_point_list(a_blob, b_blob)

    for currentNum in np.nditer(blobToChange, op_flags=['readwrite']):
        currentNum[...] = round_number(currentNum[...], blobFixedPointList)
    netToQuantize.forward(start=forwardBlob)

def detect_face_12c(net_12c_full_conv, caffe_img, min_face_size, stride,
                    multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    :param img: image in caffe style to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :param scale_factor: scale to apply for pyramid
    :param threshold: score of patch must be above this value to pass to next net
    :return:    list of rectangles after global NMS
    '''
    net_kind = 12
    rectangles = []   # list of rectangles [x11, y11, x12, y12, confidence, current_scale] (corresponding to original image)

    current_scale = find_initial_scale(net_kind, min_face_size)     # find initial scale
    caffe_img_resized = resize_image(caffe_img, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape

    while current_height > net_kind and current_width > net_kind:
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))  # switch from H x W x C to C x H x W
        # shape for input (data blob is N x C x H x W), set data
        net_12c_full_conv.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12c_full_conv.blobs['data'].data[...] = caffe_img_resized_CHW
        # run net and take argmax for prediction
        net_12c_full_conv.forward()

        # ============= Quantizing of signals ===============
        quantizeBlob(net_12c_full_conv, 'conv1', 8, 'pool1')
        quantizeBlob(net_12c_full_conv, 'pool1', 8, 'fc2-conv')
        quantizeBlob(net_12c_full_conv, 'fc2-conv', 8, 'fc3-conv')
        quantizeBlob(net_12c_full_conv, 'fc3-conv', 6, 'prob')
        # ===============================================

        out = net_12c_full_conv.blobs['prob'].data[0][1, :, :]
        # print out.shape
        out_height, out_width = out.shape

        for current_y in range(0, out_height):
            for current_x in range(0, out_width):
                # total_windows += 1
                confidence = out[current_y, current_x]  # left index is y, right index is x (starting from 0)
                if confidence >= threshold:
                    current_rectangle = [int(2*current_x*current_scale), int(2*current_y*current_scale),
                                             int(2*current_x*current_scale + net_kind*current_scale),
                                             int(2*current_y*current_scale + net_kind*current_scale),
                                             confidence, current_scale]     # find corresponding patch on image
                    rectangles.append(current_rectangle)
        if multiScale is False:
            break
        else:
            caffe_img_resized = resize_image(caffe_img_resized, scale_factor)
            current_scale *= scale_factor
            current_height, current_width, channels = caffe_img_resized.shape

    return rectangles
def cal_face_12c(net_12_cal, caffe_img, rectangles):
    '''
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''

    height, width, channels = caffe_img.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (12, 12))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_12_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_12_cal.forward()

        # ============= Quantizing of signals ===============
        quantizeBlob(net_12_cal, 'conv1', 10, 'pool1')
        quantizeBlob(net_12_cal, 'pool1', 10, 'fc2')
        quantizeBlob(net_12_cal, 'fc2', 8, 'fc3')
        quantizeBlob(net_12_cal, 'fc3', 5, 'prob')
        # ===============================================

        output = net_12_cal.blobs['prob'].data

        # output = net_12_cal.predict([cropped_caffe_img])   # predict through caffe

        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))

        result.append(cur_result)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    return result
def detect_face_24c(net_24c, caffe_img, rectangles):
    '''
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    for cur_rectangle in rectangles:

        x1 = cur_rectangle[0]
        y1 = cur_rectangle[1]
        x2 = cur_rectangle[2]
        y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (24, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_24c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_24c.blobs['data'].data[...] = caffe_img_resized_CHW
        net_24c.forward()

        # ============= Quantizing of signals ===============
        quantizeBlob(net_24c, 'conv1', 9, 'pool1')
        quantizeBlob(net_24c, 'pool1', 9, 'fc2')
        quantizeBlob(net_24c, 'fc2', 7, 'fc3')
        quantizeBlob(net_24c, 'fc3', 4, 'prob')
        # ===============================================

        prediction = net_24c.blobs['prob'].data

        confidence = prediction[0][1]

        if confidence > 0.05:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    return result
def cal_face_24c(net_24_cal, caffe_img, rectangles):
    '''
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = caffe_img.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (24, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_24_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_24_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_24_cal.forward()

        # ============= Quantizing of signals ===============
        quantizeBlob(net_24_cal, 'conv1', 10, 'pool1')
        quantizeBlob(net_24_cal, 'pool1', 10, 'fc2')
        quantizeBlob(net_24_cal, 'fc2', 9, 'fc3')
        quantizeBlob(net_24_cal, 'fc3', 5, 'prob')
        # ===============================================

        output = net_24_cal.blobs['prob'].data

        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + original_h / s_change))

        result.append(cur_result)

    return result
def detect_face_48c(net_48c, caffe_img, rectangles):
    '''
    :param caffe_img: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    result = []
    for cur_rectangle in rectangles:

        x1 = cur_rectangle[0]
        y1 = cur_rectangle[1]
        x2 = cur_rectangle[2]
        y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (48, 48))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_48c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48c.blobs['data'].data[...] = caffe_img_resized_CHW
        net_48c.forward()

        # ============= Quantizing of signals ===============
        quantizeBlob(net_48c, 'conv1', 10, 'pool1')
        quantizeBlob(net_48c, 'pool1', 10, 'conv2')
        quantizeBlob(net_48c, 'conv2', 9, 'pool2')
        quantizeBlob(net_48c, 'pool2', 9, 'fc3')
        quantizeBlob(net_48c, 'fc3', 7, 'fc4')
        quantizeBlob(net_48c, 'fc4', 6, 'prob')
        # ===============================================

        prediction = net_48c.blobs['prob'].data

        confidence = prediction[0][1]

        if confidence > 0.1:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    return result
def cal_face_48c(net_48_cal, caffe_img, rectangles):
    '''
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = caffe_img.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image
        caffe_img_resized = cv2.resize(cropped_caffe_img, (48, 48))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_48_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_48_cal.forward()

        # ============= Quantizing of signals ===============
        quantizeBlob(net_48_cal, 'conv1', 10, 'pool1')
        quantizeBlob(net_48_cal, 'pool1', 10, 'conv2')
        quantizeBlob(net_48_cal, 'conv2', 10, 'pool2')
        quantizeBlob(net_48_cal, 'pool2', 10, 'fc3')
        quantizeBlob(net_48_cal, 'fc3', 9, 'fc4')
        quantizeBlob(net_48_cal, 'fc4', 6, 'prob')
        # ===============================================

        output = net_48_cal.blobs['prob'].data

        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        total_s_change = 0
        total_x_change = 0
        total_y_change = 0

        for current_cal in range(number_of_cals):       # accumulate changes, and calculate average
            cal_label = int(indices[current_cal])   # should be number in 0~44
            if (cal_label >= 0) and (cal_label <= 8):       # decide s change
                total_s_change += 0.83
            elif (cal_label >= 9) and (cal_label <= 17):
                total_s_change += 0.91
            elif (cal_label >= 18) and (cal_label <= 26):
                total_s_change += 1.0
            elif (cal_label >= 27) and (cal_label <= 35):
                total_s_change += 1.10
            else:
                total_s_change += 1.21

            if cal_label % 9 <= 2:       # decide x change
                total_x_change += -0.17
            elif (cal_label % 9 >= 6) and (cal_label % 9 <= 8):     # ignore case when 3<=x<=5, since adding 0 doesn't change
                total_x_change += 0.17

            if cal_label % 3 == 0:       # decide y change
                total_y_change += -0.17
            elif cal_label % 3 == 2:     # ignore case when 1, since adding 0 doesn't change
                total_y_change += 0.17

        s_change = total_s_change / number_of_cals      # calculate average
        x_change = total_x_change / number_of_cals
        y_change = total_y_change / number_of_cals

        cur_result = cur_rectangle      # inherit format and last two attributes from original rectangle
        cur_result[0] = int(max(0, original_x1 - original_w * x_change / s_change))
        cur_result[1] = int(max(0, original_y1 - 1.1 * original_h * y_change / s_change))
        cur_result[2] = int(min(width, cur_result[0] + original_w / s_change))
        cur_result[3] = int(min(height, cur_result[1] + 1.1 * original_h / s_change))

        result.append(cur_result)

    return result

# ==========================================================================
total_time = 0
total_images = 0

# load and open files to read and write
for current_file in range(1, 11):

    print 'Processing file ' + str(current_file) + ' ...'

    read_file_name = './FDDB-fold/FDDB-fold-' + str(current_file).zfill(2) + '.txt'
    write_file_name = './detections/fold-' + str(current_file).zfill(2) + '-out.txt'
    write_file = open(write_file_name, "w")

    with open(read_file_name, "r") as ins:
        array = []
        for line in ins:
            array.append(line)      # list of strings

    number_of_images = len(array)

    for current_image in range(number_of_images):
        if current_image % 10 == 0:
            print 'Processing image : ' + str(current_image)
        # load image and convert to gray
        read_img_name = '/home/anson/FDDB/originalPics/' + array[current_image].rstrip() + '.jpg'
        img = cv2.imread(read_img_name)     # BGR

        min_face_size = 40
        stride = 5

        start = time.clock()

        # caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
        # caffe_image = caffe_image[:, :, (2, 1, 0)]
        img_forward = np.array(img, dtype=np.float32)
        img_forward -= np.array((104, 117, 123))

        rectangles = detect_face_12c(net_12c_full_conv, img_forward, min_face_size, stride, True)  # detect faces
        rectangles = cal_face_12c(net_12_cal, img_forward, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_24c(net_24c, img_forward, rectangles)
        rectangles = cal_face_24c(net_24_cal, img_forward, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_48c(net_48c, img_forward, rectangles)
        rectangles = globalNMS(rectangles)      # apply global NMS
        rectangles = cal_face_48c(net_48_cal, img_forward, rectangles)      # calibration

        end = time.clock()
        total_time += (end - start)
        total_images += 1

        number_of_faces = len(rectangles)
        # for rectangle in rectangles:    # draw rectangles
        #     cv2.rectangle(img, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (255, 0, 0), 2)

        # write to file
        write_file.write(array[current_image])
        write_file.write("{}\n".format( str(number_of_faces) ) )
        for i in range(number_of_faces):
            write_file.write( "{} {} {} {} {}\n".format(str(rectangles[i][0]), str(rectangles[i][1]),
                                                        str(rectangles[i][2] - rectangles[i][0]),
                                                        str(rectangles[i][3] - rectangles[i][1]),
                                                        str(rectangles[i][4])))

    write_file.close()
    print 'Average time spent on one image : ' + str(total_time / total_images) + ' s'
