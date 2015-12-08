import numpy as np
import sys
from PIL import Image
import cv2
import time
import os
from operator import itemgetter
sys.path.append('../face_detection')
from load_model_functions import *
from face_detection_functions import *

writeRangeFile = open('full_conv_blobs_ranges.txt', 'w')

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
# ==================  load models  ======================================
net_12c_full_conv, net_12_cal, net_24c, net_24_cal, net_48c, net_48_cal = \
    load_face_models()
# save each blob's max and min
net12_range = [0] * 12  # net12 has 6 blobs (data max, data min, conv1 max, ... )
net12cal_range = [0] * 12
net24_range = [0] * 12
net24cal_range = [0] * 12
net48_range = [0] * 16
net48cal_range = [0] * 16

# ==========================================================================
def print_blob_ranges():
    '''
    print all blob ranges
    '''
    print "\n======= net_12c_full_conv ======="
    for k, v in net_12c_full_conv.blobs.items():
        print (k, v.data.shape)
        if k == 'data':
            maxBlob = net12_range[0]
            minBlob = net12_range[1]
        elif k == 'conv1':
            maxBlob = net12_range[2]
            minBlob = net12_range[3]
        elif k == 'pool1':
            maxBlob = net12_range[4]
            minBlob = net12_range[5]
        elif k == 'fc2-conv':
            maxBlob = net12_range[6]
            minBlob = net12_range[7]
        elif k == 'fc3-conv':
            maxBlob = net12_range[8]
            minBlob = net12_range[9]
        elif k == 'prob':
            maxBlob = net12_range[10]
            minBlob = net12_range[11]
        print ("Max : " + str(maxBlob) + "  min : " + str(minBlob))

    print "\n======= net_12_cal ======="
    for k, v in net_12_cal.blobs.items():
        print (k, v.data.shape)
        if k == 'data':
            maxBlob = net12cal_range[0]
            minBlob = net12cal_range[1]
        elif k == 'conv1':
            maxBlob = net12cal_range[2]
            minBlob = net12cal_range[3]
        elif k == 'pool1':
            maxBlob = net12cal_range[4]
            minBlob = net12cal_range[5]
        elif k == 'fc2':
            maxBlob = net12cal_range[6]
            minBlob = net12cal_range[7]
        elif k == 'fc3':
            maxBlob = net12cal_range[8]
            minBlob = net12cal_range[9]
        elif k == 'prob':
            maxBlob = net12cal_range[10]
            minBlob = net12cal_range[11]
        print ("Max : " + str(maxBlob) + "  min : " + str(minBlob))

    print "\n======= net_24c ======="
    for k, v in net_24c.blobs.items():
        print (k, v.data.shape)
        if k == 'data':
            maxBlob = net24_range[0]
            minBlob = net24_range[1]
        elif k == 'conv1':
            maxBlob = net24_range[2]
            minBlob = net24_range[3]
        elif k == 'pool1':
            maxBlob = net24_range[4]
            minBlob = net24_range[5]
        elif k == 'fc2':
            maxBlob = net24_range[6]
            minBlob = net24_range[7]
        elif k == 'fc3':
            maxBlob = net24_range[8]
            minBlob = net24_range[9]
        elif k == 'prob':
            maxBlob = net24_range[10]
            minBlob = net24_range[11]
        print ("Max : " + str(maxBlob) + "  min : " + str(minBlob))

    print "\n======= net_24_cal ======="
    for k, v in net_24_cal.blobs.items():
        print (k, v.data.shape)
        if k == 'data':
            maxBlob = net24cal_range[0]
            minBlob = net24cal_range[1]
        elif k == 'conv1':
            maxBlob = net24cal_range[2]
            minBlob = net24cal_range[3]
        elif k == 'pool1':
            maxBlob = net24cal_range[4]
            minBlob = net24cal_range[5]
        elif k == 'fc2':
            maxBlob = net24cal_range[6]
            minBlob = net24cal_range[7]
        elif k == 'fc3':
            maxBlob = net24cal_range[8]
            minBlob = net24cal_range[9]
        elif k == 'prob':
            maxBlob = net24cal_range[10]
            minBlob = net24cal_range[11]
        print ("Max : " + str(maxBlob) + "  min : " + str(minBlob))

    print "\n======= net_48c ======="
    for k, v in net_48c.blobs.items():
        print (k, v.data.shape)
        if k == 'data':
            maxBlob = net48_range[0]
            minBlob = net48_range[1]
        elif k == 'conv1':
            maxBlob = net48_range[2]
            minBlob = net48_range[3]
        elif k == 'pool1':
            maxBlob = net48_range[4]
            minBlob = net48_range[5]
        elif k == 'conv2':
            maxBlob = net48_range[6]
            minBlob = net48_range[7]
        elif k == 'pool2':
            maxBlob = net48_range[8]
            minBlob = net48_range[9]
        elif k == 'fc3':
            maxBlob = net48_range[10]
            minBlob = net48_range[11]
        elif k == 'fc4':
            maxBlob = net48_range[12]
            minBlob = net48_range[13]
        elif k == 'prob':
            maxBlob = net48_range[14]
            minBlob = net48_range[15]
        print ("Max : " + str(maxBlob) + "  min : " + str(minBlob))

    print "\n======= net_48_cal ======="
    for k, v in net_48_cal.blobs.items():
        print (k, v.data.shape)
        if k == 'data':
            maxBlob = net48cal_range[0]
            minBlob = net48cal_range[1]
        elif k == 'conv1':
            maxBlob = net48cal_range[2]
            minBlob = net48cal_range[3]
        elif k == 'pool1':
            maxBlob = net48cal_range[4]
            minBlob = net48cal_range[5]
        elif k == 'conv2':
            maxBlob = net48cal_range[6]
            minBlob = net48cal_range[7]
        elif k == 'pool2':
            maxBlob = net48cal_range[8]
            minBlob = net48cal_range[9]
        elif k == 'fc3':
            maxBlob = net48cal_range[10]
            minBlob = net48cal_range[11]
        elif k == 'fc4':
            maxBlob = net48cal_range[12]
            minBlob = net48cal_range[13]
        elif k == 'prob':
            maxBlob = net48cal_range[14]
            minBlob = net48cal_range[15]
        print ("Max : " + str(maxBlob) + "  min : " + str(minBlob))
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
        out = net_12c_full_conv.blobs['prob'].data[0][1, :, :]
        # print out.shape
        out_height, out_width = out.shape

        for k, v in net_12c_full_conv.blobs.items():
            if k == 'data':
                if v.data.max() > net12_range[0]:
                    net12_range[0] = v.data.max()
                if v.data.min() < net12_range[1]:
                    net12_range[1] = v.data.min()
            elif k == 'conv1':
                if v.data.max() > net12_range[2]:
                    net12_range[2] = v.data.max()
                if v.data.min() < net12_range[3]:
                    net12_range[3] = v.data.min()
            elif k == 'pool1':
                if v.data.max() > net12_range[4]:
                    net12_range[4] = v.data.max()
                if v.data.min() < net12_range[5]:
                    net12_range[5] = v.data.min()
            elif k == 'fc2-conv':
                if v.data.max() > net12_range[6]:
                    net12_range[6] = v.data.max()
                if v.data.min() < net12_range[7]:
                    net12_range[7] = v.data.min()
            elif k == 'fc3-conv':
                if v.data.max() > net12_range[8]:
                    net12_range[8] = v.data.max()
                if v.data.min() < net12_range[9]:
                    net12_range[9] = v.data.min()
            elif k == 'prob':
                if v.data.max() > net12_range[10]:
                    net12_range[10] = v.data.max()
                if v.data.min() < net12_range[11]:
                    net12_range[11] = v.data.min()

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
    all_cropped_caffe_img = []

    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[original_y1:original_y2, original_x1:original_x2] # crop image
        all_cropped_caffe_img.append(cropped_caffe_img)

    if len(all_cropped_caffe_img) == 0:
        return []

    output_all = net_12_cal.predict(all_cropped_caffe_img)   # predict through caffe

    for k, v in net_12_cal.blobs.items():
        if k == 'data':
            if v.data.max() > net12cal_range[0]:
                net12cal_range[0] = v.data.max()
            if v.data.min() < net12cal_range[1]:
                net12cal_range[1] = v.data.min()
        elif k == 'conv1':
            if v.data.max() > net12cal_range[2]:
                net12cal_range[2] = v.data.max()
            if v.data.min() < net12cal_range[3]:
                net12cal_range[3] = v.data.min()
        elif k == 'pool1':
            if v.data.max() > net12cal_range[4]:
                net12cal_range[4] = v.data.max()
            if v.data.min() < net12cal_range[5]:
                net12cal_range[5] = v.data.min()
        elif k == 'fc2':
            if v.data.max() > net12cal_range[6]:
                net12cal_range[6] = v.data.max()
            if v.data.min() < net12cal_range[7]:
                net12cal_range[7] = v.data.min()
        elif k == 'fc3':
            if v.data.max() > net12cal_range[8]:
                net12cal_range[8] = v.data.max()
            if v.data.min() < net12cal_range[9]:
                net12cal_range[9] = v.data.min()
        elif k == 'prob':
            if v.data.max() > net12cal_range[10]:
                net12cal_range[10] = v.data.max()
            if v.data.min() < net12cal_range[11]:
                net12cal_range[11] = v.data.min()

    for cur_rect in range(len(rectangles)):
        cur_rectangle = rectangles[cur_rect]
        output = output_all[cur_rect]
        prediction = output[0]      # (44, 1) ndarray

        threshold = 0.1
        indices = np.nonzero(prediction > threshold)[0]   # ndarray of indices where prediction is larger than threshold

        number_of_cals = len(indices)   # number of calibrations larger than threshold

        if number_of_cals == 0:     # if no calibration is needed, check next rectangle
            result.append(cur_rectangle)
            continue

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

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
    all_cropped_caffe_img = []

    for cur_rectangle in rectangles:

        x1 = cur_rectangle[0]
        y1 = cur_rectangle[1]
        x2 = cur_rectangle[2]
        y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[y1:y2, x1:x2]     # crop image
        all_cropped_caffe_img.append(cropped_caffe_img)

    if len(all_cropped_caffe_img) == 0:
        return []

    prediction_all = net_24c.predict(all_cropped_caffe_img)   # predict through caffe

    for k, v in net_24c.blobs.items():
        if k == 'data':
            if v.data.max() > net24_range[0]:
                net24_range[0] = v.data.max()
            if v.data.min() < net24_range[1]:
                net24_range[1] = v.data.min()
        elif k == 'conv1':
            if v.data.max() > net24_range[2]:
                net24_range[2] = v.data.max()
            if v.data.min() < net24_range[3]:
                net24_range[3] = v.data.min()
        elif k == 'pool1':
            if v.data.max() > net24_range[4]:
                net24_range[4] = v.data.max()
            if v.data.min() < net24_range[5]:
                net24_range[5] = v.data.min()
        elif k == 'fc2':
            if v.data.max() > net24_range[6]:
                net24_range[6] = v.data.max()
            if v.data.min() < net24_range[7]:
                net24_range[7] = v.data.min()
        elif k == 'fc3':
            if v.data.max() > net24_range[8]:
                net24_range[8] = v.data.max()
            if v.data.min() < net24_range[9]:
                net24_range[9] = v.data.min()
        elif k == 'prob':
            if v.data.max() > net24_range[10]:
                net24_range[10] = v.data.max()
            if v.data.min() < net24_range[11]:
                net24_range[11] = v.data.min()

    for cur_rect in range(len(rectangles)):
        confidence = prediction_all[cur_rect][1]
        if confidence > 0.05:
            cur_rectangle = rectangles[cur_rect]
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
        output = net_24_cal.predict([cropped_caffe_img])   # predict through caffe

        for k, v in net_24_cal.blobs.items():
            if k == 'data':
                if v.data.max() > net24cal_range[0]:
                    net24cal_range[0] = v.data.max()
                if v.data.min() < net24cal_range[1]:
                    net24cal_range[1] = v.data.min()
            elif k == 'conv1':
                if v.data.max() > net24cal_range[2]:
                    net24cal_range[2] = v.data.max()
                if v.data.min() < net24cal_range[3]:
                    net24cal_range[3] = v.data.min()
            elif k == 'pool1':
                if v.data.max() > net24cal_range[4]:
                    net24cal_range[4] = v.data.max()
                if v.data.min() < net24cal_range[5]:
                    net24cal_range[5] = v.data.min()
            elif k == 'fc2':
                if v.data.max() > net24cal_range[6]:
                    net24cal_range[6] = v.data.max()
                if v.data.min() < net24cal_range[7]:
                    net24cal_range[7] = v.data.min()
            elif k == 'fc3':
                if v.data.max() > net24cal_range[8]:
                    net24cal_range[8] = v.data.max()
                if v.data.min() < net24cal_range[9]:
                    net24cal_range[9] = v.data.min()
            elif k == 'prob':
                if v.data.max() > net24cal_range[10]:
                    net24cal_range[10] = v.data.max()
                if v.data.min() < net24cal_range[11]:
                    net24cal_range[11] = v.data.min()

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
    all_cropped_caffe_img = []

    for cur_rectangle in rectangles:

        x1 = cur_rectangle[0]
        y1 = cur_rectangle[1]
        x2 = cur_rectangle[2]
        y2 = cur_rectangle[3]

        cropped_caffe_img = caffe_img[y1:y2, x1:x2]     # crop image
        all_cropped_caffe_img.append(cropped_caffe_img)

        prediction = net_48c.predict([cropped_caffe_img])   # predict through caffe

        for k, v in net_48c.blobs.items():
            if k == 'data':
                if v.data.max() > net48_range[0]:
                    net48_range[0] = v.data.max()
                if v.data.min() < net48_range[1]:
                    net48_range[1] = v.data.min()
            elif k == 'conv1':
                if v.data.max() > net48_range[2]:
                    net48_range[2] = v.data.max()
                if v.data.min() < net48_range[3]:
                    net48_range[3] = v.data.min()
            elif k == 'pool1':
                if v.data.max() > net48_range[4]:
                    net48_range[4] = v.data.max()
                if v.data.min() < net48_range[5]:
                    net48_range[5] = v.data.min()
            elif k == 'conv2':
                if v.data.max() > net48_range[6]:
                    net48_range[6] = v.data.max()
                if v.data.min() < net48_range[7]:
                    net48_range[7] = v.data.min()
            elif k == 'pool2':
                if v.data.max() > net48_range[8]:
                    net48_range[8] = v.data.max()
                if v.data.min() < net48_range[9]:
                    net48_range[9] = v.data.min()
            elif k == 'fc3':
                if v.data.max() > net48_range[10]:
                    net48_range[10] = v.data.max()
                if v.data.min() < net48_range[11]:
                    net48_range[11] = v.data.min()
            elif k == 'fc4':
                if v.data.max() > net48_range[12]:
                    net48_range[12] = v.data.max()
                if v.data.min() < net48_range[13]:
                    net48_range[13] = v.data.min()
            elif k == 'prob':
                if v.data.max() > net48_range[14]:
                    net48_range[14] = v.data.max()
                if v.data.min() < net48_range[15]:
                    net48_range[15] = v.data.min()

        confidence = prediction[0][1]

        if confidence > 0.3:
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
        output = net_48_cal.predict([cropped_caffe_img])   # predict through caffe

        for k, v in net_48_cal.blobs.items():
            if k == 'data':
                if v.data.max() > net48cal_range[0]:
                    net48cal_range[0] = v.data.max()
                if v.data.min() < net48cal_range[1]:
                    net48cal_range[1] = v.data.min()
            elif k == 'conv1':
                if v.data.max() > net48cal_range[2]:
                    net48cal_range[2] = v.data.max()
                if v.data.min() < net48cal_range[3]:
                    net48cal_range[3] = v.data.min()
            elif k == 'pool1':
                if v.data.max() > net48cal_range[4]:
                    net48cal_range[4] = v.data.max()
                if v.data.min() < net48cal_range[5]:
                    net48cal_range[5] = v.data.min()
            elif k == 'conv2':
                if v.data.max() > net48cal_range[6]:
                    net48cal_range[6] = v.data.max()
                if v.data.min() < net48cal_range[7]:
                    net48cal_range[7] = v.data.min()
            elif k == 'pool2':
                if v.data.max() > net48cal_range[8]:
                    net48cal_range[8] = v.data.max()
                if v.data.min() < net48cal_range[9]:
                    net48cal_range[9] = v.data.min()
            elif k == 'fc3':
                if v.data.max() > net48cal_range[10]:
                    net48cal_range[10] = v.data.max()
                if v.data.min() < net48cal_range[11]:
                    net48cal_range[11] = v.data.min()
            elif k == 'fc4':
                if v.data.max() > net48cal_range[12]:
                    net48cal_range[12] = v.data.max()
                if v.data.min() < net48cal_range[13]:
                    net48cal_range[13] = v.data.min()
            elif k == 'prob':
                if v.data.max() > net48cal_range[14]:
                    net48cal_range[14] = v.data.max()
                if v.data.min() < net48cal_range[15]:
                    net48cal_range[15] = v.data.min()

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
# load and open files to read and write
for current_file in range(1, 11):

    print 'Processing file ' + str(current_file) + ' ...'

    read_file_name = '../face_detection/FDDB-fold/FDDB-fold-' + str(current_file).zfill(2) + '.txt'

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

        caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
        caffe_image = caffe_image[:, :, (2, 1, 0)]
        img_forward = np.array(img, dtype=np.float32)
        img_forward -= np.array((104.00698793, 116.66876762, 122.67891434))

        rectangles = detect_face_12c(net_12c_full_conv, img_forward, min_face_size, stride, True)     # detect faces
        rectangles = cal_face_12c(net_12_cal, caffe_image, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_24c(net_24c, caffe_image, rectangles)
        rectangles = cal_face_24c(net_24_cal, caffe_image, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_48c(net_48c, caffe_image, rectangles)
        rectangles = globalNMS(rectangles)      # apply global NMS
        rectangles = cal_face_48c(net_48_cal, caffe_image, rectangles)      # calibration

sys.stdout = writeRangeFile
print_blob_ranges()
writeRangeFile.close()
