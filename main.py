import face_recognition
import cv2
import numpy as np
from imutils import face_utils
import argparse
import imutils
import dlib
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import numpy as np
import os

PATH = '/home/shaaran/PycharmProjects/robot_emotion/'
os.chdir('/home/shaaran/PycharmProjects/robot_emotion/')
sz = 224
arch = resnet34

def gitsearch():
# This part contains the main code.



        video_capture = cv2.VideoCapture(0) #starts the web cam if you attach it externally use 1 or 2 , use trail and error
        detector = dlib.get_frontal_face_detector()
        predict_path = '/home/shaaran/PycharmProjects/shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(predict_path)
        count = 0
        tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
        data = ImageClassifierData.from_paths(PATH, tfms=tfms)
        print(data.classes)

        learn = ConvLearner.pretrained(arch, data, precompute=True)
        print('loading requirements......')
        print('This has been made by shaaran alias devshaaran, if you are using this code anywhere for research or educational purposes, please give reference.ENJOY!')
        learn.precompute=False
        #learn.fit(1e-1, 1)
        learn.fit(1e-1, 3, cycle_len=1)
        learn.load('224_all')
        print('loading done !')


# Initialize some variablesface_locations = []

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face detection processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame, model="cnn")
            counts = 0
            counts += 1

            # Display the results
            for top, right, bottom, left in face_locations:
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                lower_red = np.array([0,0,253])
                upper_red = np.array([0,0,255])




                # Extract the region of the image that contains the face
                face_image = frame[top:bottom, left:right]
                mask = cv2.inRange(face_image, lower_red, upper_red)
                res = cv2.bitwise_and(face_image, face_image, mask=mask)

                #face_landmarks_list = face_recognition.face_landmarks(face_image)

                #for face_landmarks in face_landmarks_list:


                    #for facial_feature in facial_features:
                        #d.line(face_landmarks[facial_feature], width=5)

                    #pil_image.show()


                cv2.imshow('vid', face_image)
                cv2.imshow('res', res)
                count += 1
                cv2.imwrite('0.jpg',res)
                #cv2.imwrite((output_loc + '\\' + str(count)+ str(counts) + '.jpg'), res)

                try:

                   # learn = ConvLearner.pretrained(arch, data, precompute=True)
                    trn_tfms, val_tfms = tfms_from_model(arch, sz)
                    im = val_tfms(open_image('0.jpg'))
                    learn.precompute = False
                    preds = learn.predict_array(im[None])
                    #print(preds)
                    print(data.classes[np.argmax(preds)])

                except Exception as e:
                    print(e)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()


def rect_to_bb(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y


    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)


    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def dlibfound():

    detector = dlib.get_frontal_face_detector()
    predict_path = 'D:\Python_mytext\cv2_notes\shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predict_path)
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, image = video_capture.read()
        #image = cv2.imread(args["image"])
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped_image = image[x:x+w,y:y+h]


            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv2.imshow('cropped', cropped_image)
            cv2.imwrite('0.jpg',cropped_image)






        cv2.imshow("Output", image)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video_capture.release()

gitsearch()
