import numpy as np
import cv2
import time
from operator import itemgetter

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# ==================  load face_12c  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12c/face12c_full_conv.caffemodel'
caffe.set_mode_gpu()
net_12c_full_conv = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
# ==================  load face_12_cal  ======================================
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_12_cal/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_12_cal/face_12_cal_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
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
PRETRAINED = '/home/anson/caffe-master/models/face_24c/face_24c_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
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
PRETRAINED = '/home/anson/caffe-master/models/face_24_cal/face_24_cal_train_iter_400000.caffemodel'
caffe.set_mode_gpu()
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
PRETRAINED = '/home/anson/caffe-master/models/face_48c/face_48c_train_iter_200000.caffemodel'
caffe.set_mode_gpu()
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
PRETRAINED = '/home/anson/caffe-master/models/face_48_cal/face_48_cal_train_iter_390000.caffemodel'
caffe.set_mode_gpu()
net_48_cal = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(60, 60))
# .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>

# ==================  caffe  ======================================
def find_initial_scale(net_kind, min_face_size):
    '''
    :param net_kind: what kind of net (12, 24, or 48)
    :param min_face_size: minimum face size
    :return:    returns scale factor
    '''
    return float(min_face_size) / net_kind
def resize_image(img, scale):
    '''
    :param img: original img
    :param scale: scale factor
    :return:    resized image
    '''
    height, width, channels = img.shape
    new_height = int(height / scale)     # resized new height
    new_width = int(width / scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim)      # resized image
    return img_resized
def draw_rectangle(net_kind, img, face):
    '''
    :param net_kind: what kind of net (12, 24, or 48)
    :param img: image to draw on
    :param face: # list of info. in format [x, y, scale]
    :return:    nothing
    '''
    x = face[0]
    y = face[1]
    scale = face[2]
    original_x = int(x * scale)      # corresponding x and y at original image
    original_y = int(y * scale)
    original_x_br = int(x * scale + net_kind * scale)    # bottom right x and y
    original_y_br = int(y * scale + net_kind * scale)
    cv2.rectangle(img, (original_x, original_y), (original_x_br, original_y_br), (255,0,0), 2)
def IoU(rect_1, rect_2):
    '''
    :param rect_1: list in format [x11, y11, x12, y12, confidence, current_scale]
    :param rect_2:  list in format [x21, y21, x22, y22, confidence, current_scale]
    :return:    returns IoU ratio (intersection over union) of two rectangles
    '''
    x11 = rect_1[0]    # first rectangle top left x
    y11 = rect_1[1]    # first rectangle top left y
    x12 = rect_1[2]    # first rectangle bottom right x
    y12 = rect_1[3]    # first rectangle bottom right y
    x21 = rect_2[0]    # second rectangle top left x
    y21 = rect_2[1]    # second rectangle top left y
    x22 = rect_2[2]    # second rectangle bottom right x
    y22 = rect_2[3]    # second rectangle bottom right y
    x_overlap = max(0, min(x12,x22) -max(x11,x21))
    y_overlap = max(0, min(y12,y22) -max(y11,y21))
    intersection = x_overlap * y_overlap
    union = (x12-x11) * (y12-y11) + (x22-x21) * (y22-y21) - intersection
    return float(intersection) / union
def IoM(rect_1, rect_2):
    '''
    :param rect_1: list in format [x11, y11, x12, y12, confidence, current_scale]
    :param rect_2:  list in format [x21, y21, x22, y22, confidence, current_scale]
    :return:    returns IoM ratio (intersection over min-area) of two rectangles
    '''
    x11 = rect_1[0]    # first rectangle top left x
    y11 = rect_1[1]    # first rectangle top left y
    x12 = rect_1[2]    # first rectangle bottom right x
    y12 = rect_1[3]    # first rectangle bottom right y
    x21 = rect_2[0]    # second rectangle top left x
    y21 = rect_2[1]    # second rectangle top left y
    x22 = rect_2[2]    # second rectangle bottom right x
    y22 = rect_2[3]    # second rectangle bottom right y
    x_overlap = max(0, min(x12,x22) -max(x11,x21))
    y_overlap = max(0, min(y12,y22) -max(y11,y21))
    intersection = x_overlap * y_overlap
    rect1_area = (y12 - y11) * (x12 - x11)
    rect2_area = (y22 - y21) * (x22 - x21)
    min_area = min(rect1_area, rect2_area)
    return float(intersection) / min_area
def localNMS(rectangles):
    '''
    :param rectangles:  list of rectangles, which are lists in format [x11, y11, x12, y12, confidence, current_scale],
                        sorted from highest confidence to smallest
    :return:    list of rectangles after local NMS
    '''
    result_rectangles = rectangles[:]  # list to return
    number_of_rects = len(result_rectangles)
    threshold = 0.3     # threshold of IoU of two rectangles
    cur_rect = 0
    while cur_rect < number_of_rects - 1:     # start from first element to second last element
        rects_to_compare = number_of_rects - cur_rect - 1      # elements after current element to compare
        cur_rect_to_compare = cur_rect + 1    # start comparing with element ater current
        while rects_to_compare > 0:      # while there is at least one element after current to compare
            if (IoU(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare]) >= threshold) \
                    and (result_rectangles[cur_rect][5] == result_rectangles[cur_rect_to_compare][5]):  # scale is same

                del result_rectangles[cur_rect_to_compare]      # delete the rectangle
                number_of_rects -= 1
            else:
                cur_rect_to_compare += 1    # skip to next rectangle
            rects_to_compare -= 1
        cur_rect += 1   # finished comparing for current rectangle

    return result_rectangles
def globalNMS(rectangles):
    '''
    :param rectangles:  list of rectangles, which are lists in format [x11, y11, x12, y12, confidence, current_scale],
                        sorted from highest confidence to smallest
    :return:    list of rectangles after global NMS
    '''
    result_rectangles = rectangles[:]  # list to return
    number_of_rects = len(result_rectangles)
    threshold = 0.3     # threshold of IoU of two rectangles
    cur_rect = 0
    while cur_rect < number_of_rects - 1:     # start from first element to second last element
        rects_to_compare = number_of_rects - cur_rect - 1      # elements after current element to compare
        cur_rect_to_compare = cur_rect + 1    # start comparing with element ater current
        while rects_to_compare > 0:      # while there is at least one element after current to compare
            if IoU(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare]) >= 0.2  \
                    or ((IoM(result_rectangles[cur_rect], result_rectangles[cur_rect_to_compare]) >= threshold)
                        and (result_rectangles[cur_rect_to_compare][5] < 0.85)):  # if IoU ratio is higher than threshold
                del result_rectangles[cur_rect_to_compare]      # delete the rectangle
                number_of_rects -= 1
            else:
                cur_rect_to_compare += 1    # skip to next rectangle
            rects_to_compare -= 1
        cur_rect += 1   # finished comparing for current rectangle

    return result_rectangles
def detect_face_12c(caffe_img, min_face_size, stride, multiScale=False):
    '''
    :param img: image in caffe style to detect faces
    :param min_face_size: minimum face size to detect (in pixels)
    :param stride: stride (in pixels)
    :param multiScale: whether to find faces under multiple scales or not
    :return:    list of rectangles after global NMS
    '''
    net_kind = 12
    scale_factor = 1.414
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

        threshold = 0.05

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
def cal_face_12c(caffe_img, rectangles):
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
def detect_face_24c(caffe_img, rectangles):
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

    for cur_rect in range(len(rectangles)):
        confidence = prediction_all[cur_rect][1]
        if confidence > 0.05:
            cur_rectangle = rectangles[cur_rect]
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    return result
def cal_face_24c(caffe_img, rectangles):
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
def detect_face_48c(caffe_img, rectangles):
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

        prediction = net_24c.predict([cropped_caffe_img])   # predict through caffe
        confidence = prediction[0][1]

        if confidence > 0.3:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    return result
def cal_face_48c(caffe_img, rectangles):
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

total_time = 0
total_images = 0

# load and open files to read and write
for current_file in range(8, 11):

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

        caffe_image = np.true_divide(img, 255)      # convert to caffe style (0~1 BGR)
        caffe_image = caffe_image[:, :, (2, 1, 0)]
        img_forward = np.array(img, dtype=np.float32)
        img_forward -= np.array((104.00698793, 116.66876762, 122.67891434))

        rectangles = detect_face_12c(img_forward, min_face_size, stride, True)     # detect faces
        rectangles = cal_face_12c(caffe_image, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_24c(caffe_image, rectangles)
        rectangles = cal_face_24c(caffe_image, rectangles)      # calibration
        rectangles = localNMS(rectangles)      # apply local NMS
        rectangles = detect_face_48c(caffe_image, rectangles)
        rectangles = globalNMS(rectangles)      # apply global NMS
        rectangles = cal_face_48c(caffe_image, rectangles)      # calibration

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
