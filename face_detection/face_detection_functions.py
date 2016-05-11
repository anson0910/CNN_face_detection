import numpy as np
import cv2
import time
from operator import itemgetter
# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

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

# ====== Below functions (12cal ~ 48cal) take images in style of caffe (0~1 BGR)===
def detect_face_12c(net_12c_full_conv, img, min_face_size, stride,
                    multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    :param img: image to detect faces
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
    caffe_img_resized = resize_image(img, current_scale)      # resized initial caffe image
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

def detect_faces(nets, img_forward, caffe_image, min_face_size, stride,
                 multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    Complete flow of face cascade detection
    :param nets: 6 nets as a tuple
    :param img_forward: image in normal style after subtracting mean pixel value
    :param caffe_image: image in style of caffe (0~1 BGR)
    :param min_face_size:
    :param stride:
    :param multiScale:
    :param scale_factor:
    :param threshold:
    :return: list of rectangles
    '''
    net_12c_full_conv = nets[0]
    net_12_cal = nets[1]
    net_24c = nets[2]
    net_24_cal = nets[3]
    net_48c = nets[4]
    net_48_cal = nets[5]

    rectangles = detect_face_12c(net_12c_full_conv, img_forward, min_face_size,
                                 stride, multiScale, scale_factor, threshold)     # detect faces
    rectangles = cal_face_12c(net_12_cal, caffe_image, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_24c(net_24c, caffe_image, rectangles)
    rectangles = cal_face_24c(net_24_cal, caffe_image, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_48c(net_48c, caffe_image, rectangles)
    rectangles = globalNMS(rectangles)      # apply global NMS
    rectangles = cal_face_48c(net_48_cal, caffe_image, rectangles)      # calibration

    return rectangles

# ========== Adjusts net to take one crop of image only during test time ==========
# ====== Below functions take images in normal style after subtracting mean pixel value===
def detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size, stride,
                    multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    Adjusts net to take one crop of image only during test time
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
    caffe_img_resized = resize_image(img_forward, current_scale)      # resized initial caffe image
    current_height, current_width, channels = caffe_img_resized.shape

    # print "Shape after resizing : " + str(caffe_img_resized.shape)

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

        # threshold = 0.02
        # idx = out[:, :] >= threshold
        # out[idx] = 1
        # idx = out[:, :] < threshold
        # out[idx] = 0
        # cv2.imshow('img', out*255)
        # cv2.waitKey(0)

        # print "Shape of output after resizing " + str(caffe_img_resized.shape) + " : " + str(out.shape)

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
def cal_face_12c_net(net_12_cal, img_forward, rectangles):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''

    height, width, channels = img_forward.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (12, 12))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_12_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_12_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_12_cal.forward()

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
def detect_face_24c_net(net_24c, img_forward, rectangles):
    '''
    Adjusts net to take one crop of image only during test time
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

        cropped_caffe_img = img_forward[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (24, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_24c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_24c.blobs['data'].data[...] = caffe_img_resized_CHW
        net_24c.forward()

        prediction = net_24c.blobs['prob'].data

        confidence = prediction[0][1]

        if confidence > 0.05:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    return result
def cal_face_24c_net(net_24_cal, img_forward, rectangles):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = img_forward.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (24, 24))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_24_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_24_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_24_cal.forward()

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
def detect_face_48c_net(net_48c, img_forward, rectangles):
    '''
    Adjusts net to take one crop of image only during test time
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

        cropped_caffe_img = img_forward[y1:y2, x1:x2]     # crop image

        caffe_img_resized = cv2.resize(cropped_caffe_img, (48, 48))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_48c.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48c.blobs['data'].data[...] = caffe_img_resized_CHW
        net_48c.forward()

        prediction = net_48c.blobs['prob'].data

        confidence = prediction[0][1]

        if confidence > 0.1:
            cur_rectangle[4] = confidence
            result.append(cur_rectangle)

    result = sorted(result, key=itemgetter(4), reverse=True)    # sort rectangles according to confidence
                                                                        # reverse, so that it ranks from large to small
    return result
def cal_face_48c_net(net_48_cal, img_forward, rectangles):
    '''
    Adjusts net to take one crop of image only during test time
    :param caffe_image: image in caffe style to detect faces
    :param rectangles:  rectangles in form [x11, y11, x12, y12, confidence, current_scale]
    :return:    rectangles after calibration
    '''
    height, width, channels = img_forward.shape
    result = []
    for cur_rectangle in rectangles:

        original_x1 = cur_rectangle[0]
        original_y1 = cur_rectangle[1]
        original_x2 = cur_rectangle[2]
        original_y2 = cur_rectangle[3]
        original_w = original_x2 - original_x1
        original_h = original_y2 - original_y1

        cropped_caffe_img = img_forward[original_y1:original_y2, original_x1:original_x2] # crop image
        caffe_img_resized = cv2.resize(cropped_caffe_img, (48, 48))
        caffe_img_resized_CHW = caffe_img_resized.transpose((2, 0, 1))
        net_48_cal.blobs['data'].reshape(1, *caffe_img_resized_CHW.shape)
        net_48_cal.blobs['data'].data[...] = caffe_img_resized_CHW
        net_48_cal.forward()

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

def detect_faces_net(nets, img_forward, min_face_size, stride,
                 multiScale=False, scale_factor=1.414, threshold=0.05):
    '''
    Complete flow of face cascade detection
    :param nets: 6 nets as a tuple
    :param img_forward: image in normal style after subtracting mean pixel value
    :param min_face_size:
    :param stride:
    :param multiScale:
    :param scale_factor:
    :param threshold:
    :return: list of rectangles
    '''
    net_12c_full_conv = nets[0]
    net_12_cal = nets[1]
    net_24c = nets[2]
    net_24_cal = nets[3]
    net_48c = nets[4]
    net_48_cal = nets[5]

    rectangles = detect_face_12c_net(net_12c_full_conv, img_forward, min_face_size,
                                 stride, multiScale, scale_factor, threshold)  # detect faces
    rectangles = cal_face_12c_net(net_12_cal, img_forward, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_24c_net(net_24c, img_forward, rectangles)
    rectangles = cal_face_24c_net(net_24_cal, img_forward, rectangles)      # calibration
    rectangles = localNMS(rectangles)      # apply local NMS
    rectangles = detect_face_48c_net(net_48c, img_forward, rectangles)
    rectangles = globalNMS(rectangles)      # apply global NMS
    rectangles = cal_face_48c_net(net_48_cal, img_forward, rectangles)      # calibration

    return rectangles
