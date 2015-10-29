import cv2
import os
os.environ['http_proxy']=''
import urllib
import numpy as np


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

# face_scrub_file = '/home/anson/face_pictures/faceScrub/facescrub_actors.txt'
# save_dir = '/home/anson/face_pictures/faceScrub/images'
# save_crop_dir = '/home/anson/face_pictures/positives/positive_14'
# suspicious_file_name = '/home/anson/face_pictures/suspicious'

face_scrub_file = '/home/anson/face_pictures/faceScrub/facescrub_actresses.txt'
save_dir = '/home/anson/face_pictures/faceScrub/images_actress'
save_crop_dir = '/home/anson/face_pictures/positives/positive_15'
suspicious_file_name = '/home/anson/face_pictures/suspicious'

array = []
with open(face_scrub_file, "r") as ins:
    for line in ins:
        line = line.replace('  ', ' ')
        array.append(line)
array = array[1:]   # get rid of header

number_of_images = len(array)

print 'Total number of images : ' + str(number_of_images)

for current_image in range(52023, number_of_images):
    if current_image % 10 == 0:
        write_content = 'Processing image ' + str(current_image) + '\n'
        print write_content

    informaton = array[current_image].split()
    image_url = informaton[4]
    coordinates = informaton[5].replace(',', ' ').split()   # get coordinates

    save_image_name = save_dir + '/' + str(current_image).zfill(6) + '.jpg'
    try:
        urllib.urlretrieve(image_url, save_image_name)      # download image from url
    except:
        # print 'Failed.'
        continue
    img = cv2.imread(save_image_name)       # load image

    if img is None:
        # print 'Image ' + str(current_image) + ' is None.'
        os.remove(save_image_name)
        continue

    x1 = int(coordinates[0])
    y1 = int(coordinates[1])
    x2 = int(coordinates[2])
    y2 = int(coordinates[3])
    cropped_img = img[y1:y2, x1:x2]

    height, width, channels = cropped_img.shape

    if cropped_img is None or height == 0 or width == 0:
        # print 'Image ' + str(current_image) + ' is None.'
        continue

    file_name = save_crop_dir + "/pos15_" + str(current_image).zfill(6) + ".jpg"
    cv2.imwrite(file_name, cropped_img)

    # caffe check suspicious faces
    caffe_image = np.true_divide(img, 255)      # convert to caffe style
    caffe_image = caffe_image[:, :, (2, 1, 0)]
    caffe_cropped_face = caffe_image[y1:y2, x1:x2]
    prediction = net.predict([caffe_cropped_face])
    confidence = prediction[0][1]
    if confidence < 0.69:
        suspicious_image_name = suspicious_file_name + "/pos15_" + str(current_image).zfill(6) + ".jpg"
        cv2.imwrite(suspicious_image_name, cropped_img)     # copy suspicious image to file
        print file_name

    # cv2.imshow('test', cropped_img)
    # cv2.waitKey(0)