import numpy as np
import cv2

# load cascade
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

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

    for current_image in range(len(array)):
        # load image and convert to gray
        read_img_name = '/home/anson/FDDB/originalPics/' + array[current_image].rstrip() + '.jpg'
        img = cv2.imread(read_img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.2, 0)     # don't group faces, and see confidence by finding how many
                                                                # iterations of grouping till the face is still there
        rectList = faces[:]     # rectList changes every iteration of grouping
        number_of_faces = len(faces)
        confidences = []
        confidence_denominator = 55.0
        group_threshold = 1     # start from 3, and see how far it goes
        last_number_of_faces = len(rectList)    # store last number of faces, so that we know how many faces have disappeared

        while len(rectList) != 0:
            rectList, weights = cv2.groupRectangles(list(faces), group_threshold)
            number_of_faces_gone = last_number_of_faces - len(rectList)
            if number_of_faces_gone > 0:        # if any faces were deleted at last grouping, append confidence
                for i in range(number_of_faces_gone):
                    confidences.append( min( (group_threshold - 1) / confidence_denominator , 1))
            last_number_of_faces = len(rectList)
            group_threshold += 1

        for i in range(last_number_of_faces):   # append confidences of faces in last round
            confidences.append( int(group_threshold - 1) / confidence_denominator )

        # for i in range(number_of_faces):
        #     print str(faces[i][0]) + ' ' + str(faces[i][1]) + ' ' + str(faces[i][2]) + ' ' + str(faces[i][3]),
        #     print str(' ' + str(confidences[i]))

        # write to file
        write_file.write(array[current_image])
        write_file.write("{}\n".format( str(number_of_faces) ) )
        for i in range(number_of_faces):
            write_file.write( "{} {} {} {} {}\n".format( str(faces[i][0]), str(faces[i][1]), str(faces[i][2]), str(faces[i][3]), str(confidences[i]) ) )
    write_file.close()

cv2.waitKey(0)
cv2.destroyAllWindows()


