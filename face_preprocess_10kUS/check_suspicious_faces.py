import os
import cv2
import numpy as np

# ==================  caffe  ======================================
caffe_root = '/home/anson/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/anson/caffe-master/models/face_48/deploy.prototxt'
PRETRAINED = '/home/anson/caffe-master/models/face_48/face_48_train_iter_100000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(60, 60))
# .mean(1).mean(1) means computing mean pixel from mean.npy, resulting in (3, ) <type 'numpy.ndarray'>
# ==================  caffe  ======================================

data_base_dir = "/home/anson/face_pictures/positives"    # file containing files of pictures
suspicious_file_name = '/home/anson/face_pictures/suspicious'

check_start_file = 1
check_end_file = 4

for current_file in range(check_start_file, check_end_file + 1):
    print 'Checking file number ' + str(current_file)
    current_file_dir = data_base_dir + '/positive_' + str(current_file).zfill(2)

    image_file_list = []
    for file in os.listdir(current_file_dir):
        if file.endswith(".jpg"):
            image_file_list.append(file)
    number_of_images = len(image_file_list)

    for current_image in range(number_of_images):

        image_file_name = current_file_dir + '/' + image_file_list[current_image]
        img = cv2.imread(image_file_name)     # load image

        # caffe check suspicious faces
        caffe_image = np.true_divide(img, 255)      # convert to caffe style
        caffe_image = caffe_image[:, :, (2, 1, 0)]
        prediction = net.predict([caffe_image])
        confidence = prediction[0][1]
        if confidence < 0.5:
            suspicious_image_name = suspicious_file_name + "/pos" + str(current_file).zfill(2) + "_" + image_file_list[current_image]
            cv2.imwrite(suspicious_image_name, img)     # copy suspicious image to file
            print image_file_name

        # cv2.imshow('test', cropped_img)
        # cv2.waitKey(0)


